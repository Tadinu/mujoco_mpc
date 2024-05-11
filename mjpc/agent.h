// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MJPC_AGENT_H_
#define MJPC_AGENT_H_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>
#include <sstream>
#include <utility>

#include <absl/functional/any_invocable.h>
#include <absl/container/flat_hash_map.h>
#include <absl/strings/match.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>
#include <absl/strings/strip.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjui.h>
#include <mujoco/mjvisualize.h>
#include "mjpc/array_safety.h"
#include "mjpc/estimators/include.h"
#include "mjpc/planners/include.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"
#include "mjpc/estimators/include.h"
#include "mjpc/planners/include.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace mju = ::mujoco::util_mjpc;

// ----- agent constants ----- //
static constexpr double kMinTimeStep = 1.0e-4;
static constexpr double kMaxTimeStep = 0.1;
static constexpr double kMinPlanningHorizon = 1.0e-5;
static constexpr double kMaxPlanningHorizon = 2.5;

// maximum number of actions to plot
static constexpr int kMaxActionPlots = 25;

// figures
struct AgentPlots {
  mjvFigure action;
  mjvFigure cost;
  mjvFigure planner;
  mjvFigure timer;
};

class Agent {
 public:
  friend class AgentTest;

  // constructor
  Agent()
      : planners_(mjpc::LoadPlanners()), estimators_(mjpc::LoadEstimators()) {}
  explicit Agent(const mjModel* model, std::shared_ptr<Task> task);

  // destructor
  ~Agent() {
    if (model_) mj_deleteModel(model_);  // we made a copy in Initialize
  }

  // ----- methods ----- //

  // initialize data, settings, planners, state
  void Initialize(const mjModel* model) {
    // ----- model ----- //
    mjModel* old_model = model_;
    model_ = mj_copyModel(nullptr, model);  // agent's copy of model

    std::cout << "ACTU:" << model_->nu << std::endl;
    // check for limits on all actuators
    int num_missing = 0;
    for (int i = 0; i < model_->nu; i++) {
      std::cout << "ACTU:" << i << bool(model->actuator_ctrllimited[i]) << std::endl;
      if (!model_->actuator_ctrllimited[i]) {
        model_->actuator_ctrllimited[i] = 1;
        //num_missing++;
        printf("%s (actuator %i) missing limits\n",
              model_->names + model_->name_actuatoradr[i], i);
      }
    }
    if (num_missing > 0) {
      mju_error("Ctrl limits required for all actuators.\n");
    }

    // planner
    planner_ = GetNumberOrDefault(0, model, "agent_planner");

    // estimator
    estimator_ =
        estimator_enabled ? GetNumberOrDefault(0, model, "estimator") : 0;

    // integrator
    integrator_ =
        GetNumberOrDefault(model->opt.integrator, model, "agent_integrator");

    // planning horizon
    horizon_ = GetNumberOrDefault(0.5, model, "agent_horizon");

    // time step
    timestep_ = GetNumberOrDefault(1.0e-2, model, "agent_timestep");

    // planning steps
    steps_ = mju_max(mju_min(horizon_ / timestep_ + 1, kMaxTrajectoryHorizon), 1);

    active_task_id_ = gui_task_id;
    ActiveTask()->Reset(model);

    // initialize planner
    for (const auto& planner : planners_) {
      planner->Initialize(model_, *ActiveTask());
    }

    // initialize state
    state.Initialize(model);

    // initialize estimator
    if (reset_estimator && estimator_enabled) {
      for (const auto& estimator : estimators_) {
        estimator->Initialize(model_);
        estimator->Reset();
      }
    }

    // initialize estimator data
    ctrl.resize(model->nu);
    sensor.resize(model->nsensordata);

    // status
    plan_enabled = false;
    action_enabled = true;
    visualize_enabled = false;
    allocate_enabled = true;
    plot_enabled = true;

    // cost
    cost_ = 0.0;

    // counter
    count_ = 0;

    // names
    mju::strcpy_arr(this->planner_names_, kPlannerNames);
    mju::strcpy_arr(this->estimator_names_, kEstimatorNames);

    // estimator threads
    estimator_threads_ = estimator_enabled;

    // planner threads
    planner_threads_ =
        std::max(1, NumAvailableHardwareThreads() - 3 - 2 * estimator_threads_);

    // delete the previous model after all the planners have been updated to use
    // the new one.
    if (old_model) {
      mj_deleteModel(old_model);
    }
  }

  // allocate memory
  void Allocate() {
    // planner
    for (const auto& planner : planners_) {
      planner->Allocate();
    }

    // state
    state.Allocate(model_);

    // set status
    allocate_enabled = false;

    // cost
    terms_.resize(ActiveTask()->num_term * kMaxTrajectoryHorizon);
  }


  // reset data, settings, planners, state
  void Reset(const double* initial_repeated_action = nullptr) {
    // planner
    for (const auto& planner : planners_) {
      planner->Reset(kMaxTrajectoryHorizon, initial_repeated_action);
    }

    // state
    state.Reset();

    // estimator
    if (reset_estimator && estimator_enabled) {
      for (const auto& estimator : estimators_) {
        estimator->Reset();
      }
    }

    // cost
    cost_ = 0.0;

    // count
    count_ = 0;

    // cost
    std::fill(terms_.begin(), terms_.end(), 0.0);
  }

  // single planner iteration
  void PlanIteration(ThreadPool* pool) {
    // start agent timer
    auto agent_start = std::chrono::steady_clock::now();

    // set agent time and time step
    model_->opt.timestep = timestep_;
    model_->opt.integrator = integrator_;

    // set planning steps
    steps_ =
        mju_max(mju_min(horizon_ / timestep_ + 1, kMaxTrajectoryHorizon), 1);

    // plan
    if (!allocate_enabled) {
      // set state
      ActivePlanner().SetState(state);

      // copy the task's residual function parameters into a new object, which
      // remains constant during planning and doesn't require locking from the
      // rollout threads
      residual_fn_ = ActiveTask()->Residual();

      if (plan_enabled) {
        // planner policy
        ActivePlanner().OptimizePolicy(steps_, *pool);

        // compute time
        agent_compute_time_ =
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - agent_start)
                .count();

        // counter
        count_ += 1;
      } else {
        // rollout nominal policy
        ActivePlanner().NominalTrajectory(steps_, *pool);

        // set timers
        agent_compute_time_ = 0.0;
      }

      // release the planning residual function
      residual_fn_.reset();
    }
  }

  // call planner to update nominal policy
  void Plan(std::atomic<bool>& exitrequest, std::atomic<int>& uiloadrequest) {
    // instantiate thread pool
    ThreadPool pool(planner_threads_);

    // main loop
    while (!exitrequest.load()) {
      if (model_ && uiloadrequest.load() == 0) {
        PlanIteration(&pool);
      }
    }  // exitrequest sent -- stop planning
  }

  using StepJob =
      absl::AnyInvocable<void(Agent*, const mjModel*, mjData*)>;

  // runs a callback before the next physics step, on the physics thread
  void RunBeforeStep(StepJob job);

  // executes all the callbacks added by RunBeforeStep. should be called on the
  // physics thread
  void ExecuteAllRunBeforeStepJobs(const mjModel* model, mjData* data);

  // modify the scene, e.g. add trace visualization
  void ModifyScene(mjvScene* scn);

  // graphical user interface elements for agent and task
  void GUI(mjUI& ui);

  // task-based GUI event
  void TaskEvent(mjuiItem* it, mjData* data, std::atomic<int>& uiloadrequest,
                 int& run);

  // agent-based GUI event
  void AgentEvent(mjuiItem* it, mjData* data, std::atomic<int>& uiloadrequest,
                  int& run);

  // estimator-based GUI event
  void EstimatorEvent(mjuiItem* it, mjData* data,
                      std::atomic<int>& uiloadrequest, int& run);

  // initialize plots
  void PlotInitialize() {
    // set figures to default
    mjv_defaultFigure(&plots_.cost);
    mjv_defaultFigure(&plots_.action);
    mjv_defaultFigure(&plots_.planner);
    mjv_defaultFigure(&plots_.timer);

    // don't rescale axes
    plots_.cost.flg_extend = 0;
    plots_.action.flg_extend = 0;
    plots_.planner.flg_extend = 0;
    plots_.timer.flg_extend = 0;

    // title
    mju::strcpy_arr(plots_.cost.title, "Objective");
    mju::strcpy_arr(plots_.action.title, "Actions");
    mju::strcpy_arr(plots_.planner.title, "Agent (log10)");
    mju::strcpy_arr(plots_.timer.title, "CPU time (msec)");

    // x-labels
    mju::strcpy_arr(plots_.action.xlabel, "Time");
    mju::strcpy_arr(plots_.timer.xlabel, "Iteration");

    // y-tick number formats
    mju::strcpy_arr(plots_.cost.yformat, "%.2f");
    mju::strcpy_arr(plots_.action.yformat, "%.2f");
    mju::strcpy_arr(plots_.planner.yformat, "%.2f");
    mju::strcpy_arr(plots_.timer.yformat, "%.2f");

    // ----- colors ----- //

    // history costs
    plots_.cost.linergb[0][0] = 1.0f;
    plots_.cost.linergb[0][1] = 1.0f;
    plots_.cost.linergb[0][2] = 1.0f;

    // current line
    plots_.cost.linergb[1][0] = 1.0f;
    plots_.cost.linergb[1][1] = 0.647f;
    plots_.cost.linergb[1][2] = 0.0f;

    // policy line
    plots_.cost.linergb[2][0] = 1.0f;
    plots_.cost.linergb[2][1] = 0.647f;
    plots_.cost.linergb[2][2] = 0.0f;

    // best cost
    plots_.cost.linergb[3][0] = 1.0f;
    plots_.cost.linergb[3][1] = 1.0f;
    plots_.cost.linergb[3][2] = 1.0f;
    int num_term = ActiveTask()->num_term;
    for (int i = 0; i < num_term; i++) {
      int nclr = kNCostColors;
      // history
      plots_.cost.linergb[4 + i][0] = CostColors[i % nclr][0];
      plots_.cost.linergb[4 + i][1] = CostColors[i % nclr][1];
      plots_.cost.linergb[4 + i][2] = CostColors[i % nclr][2];

      // prediction
      plots_.cost.linergb[4 + num_term + i][0] = 0.9 * CostColors[i % nclr][0];
      plots_.cost.linergb[4 + num_term + i][1] = 0.9 * CostColors[i % nclr][1];
      plots_.cost.linergb[4 + num_term + i][2] = 0.9 * CostColors[i % nclr][2];
    }

    // history of control
    int dim_action = mju_min(model_->nu, kMaxActionPlots);

    for (int i = 0; i < dim_action; i++) {
      plots_.action.linergb[i][0] = 0.0f;
      plots_.action.linergb[i][1] = 1.0f;
      plots_.action.linergb[i][2] = 1.0f;
    }

    // best control
    for (int i = 0; i < dim_action; i++) {
      plots_.action.linergb[dim_action + i][0] = 1.0f;
      plots_.action.linergb[dim_action + i][1] = 0.0f;
      plots_.action.linergb[dim_action + i][2] = 1.0f;
    }

    // current line
    plots_.action.linergb[2 * dim_action][0] = 1.0f;
    plots_.action.linergb[2 * dim_action][1] = 0.647f;
    plots_.action.linergb[2 * dim_action][2] = 0.0f;

    // policy line
    plots_.action.linergb[2 * dim_action + 1][0] = 1.0f;
    plots_.action.linergb[2 * dim_action + 1][1] = 0.647f;
    plots_.action.linergb[2 * dim_action + 1][2] = 0.0f;

    // history of agent compute time
    plots_.timer.linergb[0][0] = 1.0f;
    plots_.timer.linergb[0][1] = 1.0f;
    plots_.timer.linergb[0][2] = 1.0f;

    // x-tick labels
    plots_.cost.flg_ticklabel[0] = 0;
    plots_.action.flg_ticklabel[0] = 0;
    plots_.planner.flg_ticklabel[0] = 0;
    plots_.timer.flg_ticklabel[0] = 0;

    // legends

    // grid sizes
    plots_.cost.gridsize[0] = 3;
    plots_.cost.gridsize[1] = 3;
    plots_.action.gridsize[0] = 3;
    plots_.action.gridsize[1] = 3;
    plots_.planner.gridsize[0] = 3;
    plots_.planner.gridsize[1] = 3;
    plots_.timer.gridsize[0] = 3;
    plots_.timer.gridsize[1] = 3;

    // initialize
    for (int j = 0; j < 20; j++) {
      for (int i = 0; i < mjMAXLINEPNT; i++) {
        plots_.planner.linedata[j][2 * i] = static_cast<float>(-i);
        plots_.timer.linedata[j][2 * i] = static_cast<float>(-i);

        // colors
        if (j == 0) continue;
        plots_.planner.linergb[j][0] = CostColors[j][0];
        plots_.planner.linergb[j][1] = CostColors[j][1];
        plots_.planner.linergb[j][2] = CostColors[j][2];

        plots_.timer.linergb[j][0] = CostColors[j][0];
        plots_.timer.linergb[j][1] = CostColors[j][1];
        plots_.timer.linergb[j][2] = CostColors[j][2];
      }
    }
  }

  // reset plot data to zeros
  void PlotReset();

  // plot current information
  void Plots(const mjData* data, int shift);

  // return horizon (continuous time)
  double Horizon() const;

  // render plots
  void PlotShow(mjrRect* rect, mjrContext* con);

  // returns all task names, joined with '\n' characters
  std::string GetTaskNames() const { return task_names_; }
  int GetTaskIdByName(std::string_view name) const {
    for (int i = 0; i < tasks_.size(); i++) {
      if (absl::EqualsIgnoreCase(name, tasks_[i]->Name())) {
        return i;
      }
    }
    return -1;
  }
  std::string GetTaskXmlPath(int id) const { return tasks_[id]->XmlPath(); }

  // load the latest task model, based on GUI settings
  struct LoadModelResult {
    UniqueMjModel model{nullptr, mj_deleteModel};
    std::string error;
  };
  LoadModelResult LoadModel() const;

  // Sets a custom model (not from the task), to be returned by the next
  // call to LoadModel. Passing nullptr model clears the override and will
  // return the normal task's model.
  void OverrideModel(UniqueMjModel model = {nullptr, mj_deleteModel});

  mjpc::Planner& ActivePlanner() const { return *planners_[planner_]; }
  mjpc::Estimator& ActiveEstimator() const { return *estimators_[estimator_]; }
  int ActiveEstimatorIndex() const { return estimator_; }
  double ComputeTime() const { return agent_compute_time_; }
  Task* ActiveTask() const { return tasks_[active_task_id_].get(); }
  // a residual function that can be used from trajectory rollouts. must only
  // be used from trajectory rollout threads (no locking).
  const ResidualFn* PlanningResidual() const {
    return residual_fn_.get();
  }
  bool IsPlanningModel(const mjModel* model) const {
    return model == model_;
  }
  int PlanSteps() const { return steps_; }
  int GetActionDim() const { return model_->nu; }
  mjModel* GetModel() { return model_; }
  const mjModel* GetModel() const { return model_; }

  void SetTaskList(std::vector<std::shared_ptr<Task>> tasks) {
    tasks_ = std::move(tasks);
    std::ostringstream concatenated_task_names;
    for (const auto& task : tasks_) {
      concatenated_task_names << task->Name() << '\n';
    }
    mju::strcpy_arr(task_names_, concatenated_task_names.str().c_str());
  }
  void SetState(const mjData* data);
  void SetTaskByIndex(int id) { active_task_id_ = id; }
  // returns param index, or -1 if not found.
  int SetParamByName(std::string_view name, double value);
  // returns param index, or -1 if not found.
  int SetSelectionParamByName(std::string_view name, std::string_view value);
  // returns weight index, or -1 if not found.
  int SetWeightByName(std::string_view name, double value);
  // returns mode index, or -1 if not found.
  int SetModeByName(std::string_view name);

  std::vector<std::string> GetAllModeNames() const;
  std::string GetModeName() const;

  // threads
  int planner_threads() const { return planner_threads_;}
  int estimator_threads() const { return estimator_threads_;}

  // status flags, logically should be bool, but mjUI needs int pointers
  int plan_enabled;
  int action_enabled;
  int visualize_enabled;
  int allocate_enabled;
  int plot_enabled;
  int gui_task_id = 0;

  // state
  mjpc::State state;

  // estimator
  std::vector<double> sensor;
  std::vector<double> ctrl;
  bool reset_estimator = true;
  bool estimator_enabled = false;

 private:
  // model
  mjModel* model_ = nullptr;

  UniqueMjModel model_override_ = {nullptr, mj_deleteModel};

  // integrator
  int integrator_;

  // planning horizon (continuous time)
  double horizon_;

  // planning steps (number of discrete timesteps)
  int steps_;

  // time step
  double timestep_;

  std::vector<std::shared_ptr<Task>> tasks_;
  int active_task_id_ = 0;

  // residual function for the active task, updated once per planning iteration
  std::unique_ptr<ResidualFn> residual_fn_;

  // planners
  std::vector<std::unique_ptr<mjpc::Planner>> planners_;
  int planner_;

  // estimators
  std::vector<std::unique_ptr<mjpc::Estimator>> estimators_;
  int estimator_;

  // task queue for RunBeforeStep
  std::mutex step_jobs_mutex_;
  std::deque<StepJob> step_jobs_;

  // timing
  double agent_compute_time_;
  double rollout_compute_time_;

  // objective
  double cost_;
  std::vector<double> terms_;

  // planning iterations counter
  std::atomic_int count_;

  // names
  char task_names_[1024];
  char planner_names_[1024];
  char estimator_names_[1024];

  // plots
  AgentPlots plots_;

  // max threads for planning
  int planner_threads_;

  // max threads for estimation
  int estimator_threads_;
};

}  // namespace mjpc

#endif  // MJPC_AGENT_H_
