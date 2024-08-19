#include "mjpc/planners/fabrics/include/fab_planner.h"

#include <memory>

#include "mjpc/planners/fabrics/include/fab_param_tuner.h"

// Ctor & Dtor defined here instead of in header for param_tuner_ unique ptr to be acked even with forward
// declaration
FabPlanner::FabPlanner() = default;
FabPlanner::~FabPlanner() = default;

void FabPlanner::Initialize(mjModel* model, const mjpc::Task& task) {
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

  // Init task fabrics
  if (task.IsFabricsSupported()) {
    InitTaskFabrics();
  }
}

void FabPlanner::InitTaskFabrics() {
  // NOTE: This can be invoked amid a run as "Tune" is toggled (to switch the setup), so requiring a lock
  const FabSharedMutexLock lock(policy_mutex_);
  // 1- Config
  config_ = tuning_on_ ? FabPlannerConfig::get_symbolic_config()
                       : task_->GetFabricsConfig(task_->IsGoalFixed() && task_->AreObstaclesFixed());

  // 2- Robot, resetting [vars_, geometry_, target_velocity_] here-in!
  init_robot(dim_action_, task_->URDFPath(), task_->GetBaseBodyName(), task_->GetEndtipNames(), config_);

  // 3- Goal
  FabGoalComposition goal;
  for (const auto& subgoal : task_->GetSubGoals()) {
    goal.add_sub_goal(subgoal);
  }

  // 4- Add geometry + energy components + [goal]
  set_components(task_->GetCollisionLinkNames(), task_->GetSelfCollisionNamePairs(), {}, goal,
                 task_->GetJointLimits() /* TODO: Fetch from RobotURDFModel()->joint_map*/,
                 task_->GetStaticObstaclesNum(), task_->GetDynamicObstaclesNum(), 0 /*cuboid_obstacles_num*/,
                 task_->GetPlaneConstraintsNum(), task_->GetDynamicObstaclesDim());

  // 5- Concretize, calculating [xddot] + composing [cafunc_] based on it
  concretize(FAB_USE_ACTUATOR_VELOCITY ? FabControlMode::VEL : FabControlMode::ACC, 0.01);

  // 6- Param tuner (NOTE: always inited at last for using a fully-inited FabPlanner)
  param_tuner_ = std::make_unique<FabParamTuner>();
  param_tuner_->set_planner(this);
}

void FabPlanner::OptimizePolicy(int horizon, mjpc::ThreadPool& pool) {
  // get nominal trajectory
  this->NominalTrajectory(horizon, pool);

  const FabSharedMutexLock lock(policy_mutex_);
  // NOTE: [Tune] takes more priority over [Plan], which could proceed upon [Tune] finish
  // Tune
  static bool last_tuning_on = false;
  bool tuning_off_then_on = (!last_tuning_on && tuning_on_);
  last_tuning_on = tuning_on_;
  if (tuning_on_ && !param_tuner_->tuning_finished()) {
    // Fetch last trial tuning result
    param_tuner_->fetch_tune_results();

    // Start next trial tuning
    param_tuner_->tune(tuning_off_then_on);
  }
  // Plan
  else {
    plan();
  }
}