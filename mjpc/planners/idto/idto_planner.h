#pragma once

#include <mujoco/mujoco.h>

#include <cassert>
#include <chrono>
#include <map>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Drake
#include <drake/geometry/meshcat.h>
#include <drake/geometry/scene_graph.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/plant/multibody_plant_config_functions.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram_builder.h>

#include "mjpc/planners/idto/idto_common.h"
#include "mjpc/planners/idto/idto_mpc.h"
#include "mjpc/planners/idto/idto_yaml_config.h"
#include "mjpc/planners/idto/optimizer/problem_definition.h"
#include "mjpc/planners/idto/optimizer/trajectory_optimizer.h"
#include "mjpc/planners/planner.h"
#include "mjpc/utilities.h"

using drake::geometry::Meshcat;
using drake::geometry::SceneGraph;
using drake::multibody::AddMultibodyPlant;
using drake::multibody::MultibodyPlant;
using drake::multibody::MultibodyPlantConfig;
using drake::multibody::Parser;
using drake::systems::DiagramBuilder;

using idto::optimizer::ConvergenceReason;
using idto::optimizer::DecodeConvergenceReasons;
using idto::optimizer::GradientsMethod;
using idto::optimizer::LinesearchMethod;
using idto::optimizer::ProblemDefinition;
using idto::optimizer::ScalingMethod;
using idto::optimizer::SolverFlag;
using idto::optimizer::SolverMethod;
using idto::optimizer::SolverParameters;
using idto::optimizer::TrajectoryOptimizer;
using idto::optimizer::TrajectoryOptimizerSolution;
using idto::optimizer::TrajectoryOptimizerStats;

using Eigen::MatrixXd;
using Eigen::VectorXd;

class IdtoPlanner : public mjpc::Planner {
public:
  IdtoPlanner() = default;
  ~IdtoPlanner() override = default;

  /**
   * Solve the optimization problem, as defined by the parameters in the given
   * YAML file.
   *
   * @param options YAML options, incluidng cost function definition, solver
   * parameters, etc.
   * @return TrajectoryOptimizerSolution<double> the optimal trajectory
   */
  TrajectoryOptimizerSolution<double> SolveTrajectoryOptimization(const IdtoPlannerConfig& options) const;

  /**
   * Use the optimizer as an MPC controller in simulation.
   *
   * @param options YAML options, incluidng cost function definition, solver
   * parameters, etc.
   */
  void RunMPC(const IdtoPlannerConfig& options) const;

  /**
   * Set an optimization problem from example options which were loaded from
   * YAML.
   *
   * @param options parameters loaded from yaml
   * @param plant model of the system that we're optimizing over
   * @param opt_prob the problem definition (cost, initital state, etc)
   */
  void SetProblemDefinition(const IdtoPlannerConfig& options, const MultibodyPlant<double>& plant,
                            ProblemDefinition* opt_prob) const;

  /**
   * Set solver parameters (used to pass options to the optimizer)
   * from example options (loaded from a YAML file).
   *
   * @param options parameters loaded from yaml
   * @param solver_params parameters for the optimizer that we'll set
   */
  void SetSolverParameters(const IdtoPlannerConfig& options, SolverParameters* solver_params) const;

  /**
   * Normalize quaternions in the given sequence of generalized positions. This
   * is useful for, for example, ensuring that the reference and initial guess
   * contain valid quaternions.
   *
   * @param plant model of the system that we're optimizing over
   * @param q sequence of generalized positions, including quaternion DoFs, that
   * we'll normalize
   */
  void NormalizeQuaternions(const MultibodyPlant<double>& plant, std::vector<VectorXd>* q) const;

  /**
   * Normalize quaternion in the given vector of generalized positions.
   *
   * @param plant model of the system that we're optimizing over
   * @param q vector of generalized positions, including quaternion DoFs, that
   * we'll normalize
   */
  void NormalizeQuaternions(const MultibodyPlant<double>& plant, VectorXd* q) const;
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
  void Initialize(mjModel* model, const mjpc::Task& task) override;

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

    const IdtoSharedMutexLock lock(policy_mutex_);
    Plan();
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

  void StartControl();
  void Plan() {}

private:
  /**
   * Create a MultibodyPlant model of the system that we're optimizing over.
   * This is the only method that needs to be overwritten to specialize to
   * different systems.
   *
   * @param plant the MultibodyPlant that we'll add the system to.
   */
  virtual void CreatePlantModel(MultibodyPlant<double>*) const {}

  /**
   * Create a MultibodyPlant model of the system to use for simulation (i.e., to
   * test MPC). The default behavior is to use the same model that we use for
   * optimization.
   *
   * @param plant the MultibodyPlant that we'll add the system to.
   */
  virtual void CreatePlantModelForSimulation(MultibodyPlant<double>* plant) const { CreatePlantModel(plant); }

  /**
   * Play back the given trajectory on the Drake visualizer
   *
   * @param q sequence of generalized positions defining the trajectory
   * @param time_step time step (seconds) for the discretization
   */
  void PlayBackTrajectory(const std::vector<VectorXd>& q, const double time_step) const;

  /**
   * Return a vector that interpolates linearly between q_start and q_end.
   * Useful for setting initial guesses and target trajectories.
   *
   * @param start initial vector
   * @param end final vector
   * @param N number of elements in the linear interpolation
   * @return std::vector<VectorXd> vector that interpolates between start and
   * end.
   */
  std::vector<VectorXd> MakeLinearInterpolation(const VectorXd& start, const VectorXd& end, int N) const {
    std::vector<VectorXd> result;
    double lambda = 0;
    for (int i = 0; i < N; ++i) {
      lambda = i / (N - 1.0);
      result.push_back((1 - lambda) * start + lambda * end);
    }
    return result;
  }

private:
  // idto
  IdtoPlannerConfigPtr configs_ = nullptr;
  std::shared_ptr<Meshcat> meshcat_ = nullptr;

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

using IdtoPlannerPtr = std::shared_ptr<IdtoPlanner>;
