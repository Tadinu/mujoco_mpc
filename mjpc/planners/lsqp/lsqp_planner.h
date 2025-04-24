#pragma once

#include <cassert>
#include <memory>
#include <shared_mutex>

#include <mujoco/mujoco.h>

// mjpc
#include "mjpc/core/mjpc_common.h"
#include "mjpc/planners/lsqp/lsqp_solver.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/cross_entropy/planner.h"
#include "mjpc/tasks/lsqp/lsqp.h"
#include "mjpc/utilities.h"

namespace mjpc {
class Lsqp;

class LsqpPlanner : public Planner {
public:
  LsqpPlanner() = default;
  ~LsqpPlanner() override = default;

  explicit LsqpPlanner(const MjOwnerAppType type, std::unique_ptr<CrossEntropyPlanner> delegate = nullptr) :
    Planner(type),
    cem_delegate_(std::move(delegate)) {
  }

  // Init task-specific LSQP (configuration, subtasks, etc.)
  void InitTaskLsqp(const mjModel* model, const mjData* data);
  std::vector<double> LsqpControl(double* policy_action = nullptr, mjData* data = nullptr,
                                  const LsqpSolverPtr& solver = nullptr);

  // =========================================================================================================
  // MJPC-PLANNER IMPL --
  //
  mjModel* model_ = nullptr;
  // mjData is either accessed through [lsqp_task_->data_], which is already updated in [Task::TransitionLocked()],
  // BUT ONLY if it is invoked in child task's TransitionLocked()
  // OR provided by caller, eg as running in a thread of rollouts
  Lsqp* lsqp_task_ = nullptr;
  std::unique_ptr<CrossEntropyPlanner> cem_delegate_ = nullptr;

  // initialize data and settings
  void Initialize(mjModel* model, const Task& task) override;

  void Allocate() override;

  // reset memory to zeros
  void Reset(int horizon, const double* initial_repeated_action = nullptr) override {
    if (cem_delegate_) {
      cem_delegate_->Reset(horizon, initial_repeated_action);
    }
  }

  void SetState(const State& state) override {
    if (cem_delegate_) {
      cem_delegate_->SetState(state);
    }
  }

  const Trajectory* BestTrajectory() override {
    return cem_delegate_ ? cem_delegate_->BestTrajectory() : nullptr;
  }

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override;

  void ClearTrace() override {
  }

  // planner-specific GUI elements
  void GUI(mjUI& ui) override {
    if (cem_delegate_) {
      cem_delegate_->GUI(ui);
    }
  }

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift, int timer_shift, int planning,
             int* shift) override {
    if (cem_delegate_) {
      cem_delegate_->Plots(fig_planner, fig_timer, planner_shift, timer_shift, planning, shift);
    }
  }

  // return number of parameters optimized by planner
  int NumParameters() override {
    return cem_delegate_ ? cem_delegate_->NumParameters() : 0;
  }

  // optimize nominal policy
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override {
    if (cem_delegate_) {
      cem_delegate_->NominalTrajectory(horizon, pool);
    }
  }

  // set action from policy
  void ActionFromPolicy(double* action, const double* state, double time, bool use_previous) override;

protected:
  // mjpc
  mutable std::shared_mutex mutex_;
};

using LsqpPlannerPtr = std::shared_ptr<LsqpPlanner>;
} // end namespace mjpc