#pragma once

#include <mujoco/mujoco.h>

#include <Eigen/Core>
#include <cassert>
#include <chrono>
#include <map>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// mjpc
#include "mjpc/planners/cio/cio_world.h"
#include "mjpc/planners/planner.h"
#include "mjpc/utilities.h"

class CIOPlanner : public mjpc::Planner {
public:
  CIOPlanner() = default;
  ~CIOPlanner() override = default;
  // =========================================================================================================
  // MJPC-PLANNER IMPL --
  //
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  mjpc::Task* task_ = nullptr;

  /**
   * initialize data and settings, either solving a single trajectory optimization problem or
   * running MPC, as specified by the options in the YAML file.
   *
   * @param options_file YAML file containing cost function definition, solver
   * parameters, etc., with fields as defined in yaml_config.h.
   * @param test Flag for whether this is being run as a unit test. If set to
   * true, some of the options are overwritten for simplicity:
   *   - mpc = false
   *   - max_iters = 10
   *   - save_solver_stats_csv = false
   *   - play_target_trajectory = false
   *   - play_initial_guess = false
   *   - play_optimal_trajectory = false
   *   - num_threads = 1;
   **/
  void Initialize(mjModel* model, const mjpc::Task& task) override {}

  void Allocate() override {
    trajectory_->Initialize(dim_state_, dim_action_, task_->num_residual, task_->num_trace, 1);
    trajectory_->Allocate(1);
  }

  // reset memory to zeros
  void Reset(int horizon, const double* initial_repeated_action = nullptr) override {}

  void SetState(const mjpc::State& state) override {}

  const mjpc::Trajectory* BestTrajectory() override { return trajectory_.get(); }

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override {}

  void ClearTrace() override {
    const std::shared_lock<std::shared_mutex> lock(policy_mutex_);
    trajectory_->trace.clear();
  }

  // planner-specific GUI elements
  void GUI(mjUI& ui) override {}

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift, int timer_shift, int planning,
             int* shift) override {}

  // return number of parameters optimized by planner
  int NumParameters() override { return 0; }

  // optimize nominal policy
  void OptimizePolicy(int horizon, mjpc::ThreadPool& pool) override {
    // get nominal trajectory
    this->NominalTrajectory(horizon, pool);
  }

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, mjpc::ThreadPool& pool) override {}

  // set action from policy
  void ActionFromPolicy(double* action, const double* state, double time, bool use_previous) override {
    const std::shared_lock<std::shared_mutex> lock(policy_mutex_);

    // WAIT ACTION TO BE COMPUTED
    if (action_.empty()) {
      return;
    }

    // Clear [action_]
    action_.clear();

    // Clamp controls on outputted [action]
    mjpc::Clamp(action, model_->actuator_ctrlrange, model_->nu);
  }

private:
  // cio
  CIOWorldPtr cio_world_ = nullptr;

  // mjpc
  std::shared_ptr<mjpc::Trajectory> trajectory_ = nullptr;
  int dim_state_ = 0;             // state
  int dim_state_derivative_ = 0;  // state derivative
  int dim_action_ = 0;            // action
  int dim_sensor_ = 0;            // output (i.e., all sensors)
  int dim_max_ = 0;               // maximum dimension
  mutable std::shared_mutex policy_mutex_;
  // [action_] is shared among policy motion planning threads.
  // NOTE: Using type as vector of primitive, CaSX is unclear why not well synch-protected yet.
  std::vector<double> action_;
};
