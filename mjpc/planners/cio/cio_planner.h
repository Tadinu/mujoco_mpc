#pragma once

#include <mujoco/mujoco.h>

#include <Eigen/Core>
#include <cassert>
#include <chrono>
#include <map>
#include <memory>
#include <shared_mutex>
#include <vector>

// LBFGSB
#include <LBFGSB.h>

// mjpc
#include "mjpc/planners/cio/cio_common.h"
#include "mjpc/planners/cio/cio_util.h"
#include "mjpc/planners/cio/cio_world.h"
#include "mjpc/planners/planner.h"
#include "mjpc/utilities.h"

class CIOptimizer {
public:
  static constexpr int START_STAGE = 0;
  void update(int stage) {
    stage_idx_ = stage;
    goals_ = std::vector<CIOGoal>{{.pose = CIOPose(Eigen::Vector3d(5.0, 20.0, 5.0))}};
    double radius = 5.0;
    const CIOObjectPtr manip_obj = std::make_shared<CIOCuboid>(
        Eigen::Vector3d::Zero(), Eigen::Vector3d{1.0, 2.0, 3.0}, Eigen::Quaterniond::Identity());
    const auto finger0 = std::make_shared<CIOSphere>(0, 5.0, CIOPose(), Eigen::Vector3d::Zero());
    const auto finger1 = std::make_shared<CIOSphere>(1, 5.0, CIOPose(), Eigen::Vector3d::Zero());

    // initial contact information
    const CIOContactMap contact_states = {
        {std::static_pointer_cast<CIOObject>(finger0),
         {CIOContact{.f = Eigen::Vector3d::Zero(), .ro = Eigen::Vector3d(-7, -7, -7), .c = 0.5}}},
        {std::static_pointer_cast<CIOObject>(finger1),
         {CIOContact{.f = Eigen::Vector3d::Zero(), .ro = Eigen::Vector3d(7, -7, -7), .c = 0.5}}}};

    // Create world
    world_ = std::make_shared<CIOWorld>(manip_obj, std::vector{finger0, finger1}, contact_states);

    // Calculate [initials_]
    if (START_STAGE == stage_idx_) {
      std::vector<double> init;
      auto trajs = world_->stationary_trajs();
      for (auto& traj : trajs) {
        for (auto& traj_i : traj) {
          traj_i.add_noise();
          const auto traj_i_data = traj_i.data();
          std::copy(traj_i_data.begin(), traj_i_data.end(), std::back_inserter(init));
        }
      }

      initials_ = Eigen::VectorXd::Map(init.data(), init.size());
    }
  }

  Eigen::VectorXd& initials() { return initials_; }

  double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    return world_->total_cost(stage_idx_, goals_);
  }

  void optimize() {
    LBFGSpp::LBFGSBParam<double> param;
    LBFGSpp::LBFGSBSolver<double> solver(param);

    // Optimize
    for (auto i = START_STAGE; i < world_->config_.stage_weights.size(); ++i) {
      // Update [world_, goals, initials_, stage_idx_]
      update(i);

      // Variable bounds
      const int n = initials_.size();
      Eigen::VectorXd lb = Eigen::VectorXd::Constant(n, 0);  // lower
      Eigen::VectorXd ub = Eigen::VectorXd::Constant(n, 1);  // upper

      // Invoke operator(), optimizing batch of [stage_]
      double fx;
      int niter = solver.minimize(*this, initials(), fx, lb, ub);

      std::cout << niter << " iterations" << std::endl;
      // std::cout << "x = \n" << x.transpose() << std::endl;
      std::cout << "f(x) = " << fx << std::endl;
      std::cout << "grad = " << solver.final_grad().transpose() << std::endl;
      std::cout << "projected grad norm = " << solver.final_grad_norm() << std::endl;
    }
  }

private:
  // cio
  int stage_idx_ = 0;
  std::vector<CIOGoal> goals_;
  CIOWorldPtr world_ = nullptr;
  Eigen::VectorXd initials_;
};

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
  void Initialize(mjModel* model, const mjpc::Task& task) override {
    task_ = const_cast<mjpc::Task*>(&task);
    model_ = model;
    data_ = task_->data_;

    // dimensions
    dim_state_ = model->nq + model->nv + model->na;     // state dimension
    dim_state_derivative_ = 2 * model->nv + model->na;  // state derivative dimension
    dim_action_ = task.GetActionDim();                  // action dimension
    dim_sensor_ = model->nsensordata;                   // number of sensor values
    dim_max_ = std::max({dim_state_, dim_state_derivative_, dim_action_, model->nuser_sensor});

    if (trajectory_) {
      trajectory_->Reset(0);
    } else {
      trajectory_ = std::make_shared<mjpc::Trajectory>();
    }

    // Init task CIO
    if (task.IsCIOSupported()) {
      InitTaskCIO();
    }
  }

  void InitTaskCIO() {}

  void plan() {
    // Loop optimizing over multiple stages
    optimizer_.optimize();
  }

  // =========================================================================================================
  // MJPC-PLANNER IMPL --
  //
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
  CIOptimizer optimizer_;

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
