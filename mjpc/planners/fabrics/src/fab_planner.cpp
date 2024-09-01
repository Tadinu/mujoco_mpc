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
  config_ = tuning_on_ ? FabPlannerConfig::get_symbolic_config() : task_->GetFabricsConfig();

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
                 task_->GetPlaneConstraintsNum(), task_->GetDynamicObstaclesDimension());

  // 5- Concretize, calculating [xddot] + composing [cafunc_] based on it
  concretize(FAB_USE_ACTUATOR_VELOCITY ? FabControlMode::VEL : FabControlMode::ACC, 0.01);

  // 6- Param tuner (NOTE: always inited at last for using a fully-inited FabPlanner)
  if (tuning_on_) {
    param_tuner_ = std::make_unique<FabParamTuner>();
    param_tuner_->set_planner(this);
  } else {
    param_tuner_ = nullptr;
  }
}

void FabPlanner::SetGoalArguments() {
  // [X-space goals] & [Weights of goals]
  const auto sub_goals = task_->GetSubGoals();
  for (auto i = 0; i < sub_goals.size(); ++i) {
    const auto& sub_goal = sub_goals[i];
    const auto i_str = std::to_string(i);
    if (task_->IsGoalFixed()) {
      arguments_.insert_or_assign("x_goal_" + i_str, sub_goal->cfg_.desired_position);
    } else {
      arguments_.insert_or_assign("x_ref_goal_" + i_str + "_leaf", sub_goal->cfg_.desired_position);
      arguments_.insert_or_assign("xdot_ref_goal_" + i_str + "_leaf", sub_goal->cfg_.desired_vel);
      arguments_.insert_or_assign("xddot_ref_goal_" + i_str + "_leaf", sub_goal->cfg_.desired_acc);
    }
    arguments_.insert_or_assign("weight_goal_" + i_str, sub_goal->cfg_.weight);
  }
}

void FabPlanner::SetConstraintArguments() {
  for (auto i = 0; i < task_->GetPlaneConstraintsNum(); ++i) {
    arguments_.insert_or_assign("constraint_" + std::to_string(i), std::vector<double>{0, 0, 1, 0.0});
  }
}

void FabPlanner::SetCollisionArguments() {
  for (const auto& [link_name, link_size] : task_->GetCollisionLinkProps()) {
    arguments_["radius_body_" + link_name] = link_size;
  }
}

std::string FabPlanner::GetObstaclePropName(const char* prefix, int idx) const {
  return (task_->AreObstaclesFixed() ? prefix : (std::string(prefix) + "dynamic_")) + std::to_string(idx);
}

void FabPlanner::SetObstacleArguments() {
  const auto obstacle_statesX = task_->GetObstacleStatesX();
  const bool dynamic_obstacles = !task_->AreObstaclesFixed();

  const int dim = task_->GetObstaclesDim();
  std::vector obst_pos(dim, 0.);
  std::vector obst_vel(dim, 0.);
  std::vector obst_acc(dim, 0.);
  for (auto i = 0; i < obstacle_statesX.size(); ++i) {
    const auto& obstacle_i = obstacle_statesX[i];
    arguments_[GetObstaclePropName("radius_obst_", i)] = FAB_OBSTACLE_SIZE_SCALE * obstacle_i.size_[0];

    // [obst_pos]
    memcpy(obst_pos.data(), obstacle_i.pos_.data(), dim * sizeof(double));
    arguments_[GetObstaclePropName("x_obst_", i)] = obst_pos;

    if (dynamic_obstacles) {
      // [obst_vel]
      memcpy(obst_vel.data(), obstacle_i.vel_.data(), dim * sizeof(double));
      arguments_[GetObstaclePropName("xdot_obst_", i)] = obst_vel;

      // [obst_acc]
      memcpy(obst_acc.data(), obstacle_i.acc_.data(), dim * sizeof(double));
      arguments_[GetObstaclePropName("xddot_obst_", i)] = obst_acc;
    }
  }
}

void FabPlanner::SetTuningArguments(const FabParamWeightDict& params) {
#if FAB_VERIFY_TUNED_PARAMS
  static std::map<const char*, bool> fetched_params_names;
  fetched_params_names.clear();
#endif
  const auto fetch_param = [&params](const char* param_name) {
    if (params.contains(param_name)) {
#if FAB_VERIFY_TUNED_PARAMS
      fetched_params_names[param_name] = true;
#endif
      return params.at(param_name);
    }
    throw FabError::customized("SetTuningArguments", "Undefined param: " + std::string(param_name));
  };

  const auto fconstraint_prop_name = [](const char* prefix, const std::string& link_name, const int i) {
    return std::string(prefix) + link_name + "_constraint_" + std::to_string(i);
  };
  for (const auto& link_name : task_->GetCollisionLinkNames()) {
    for (auto i = 0; i < task_->GetObstacleStatesX().size(); ++i) {
      // LINK OBSTACLE LEAF
      const auto link_name_leaf_str = link_name + "_leaf";
      arguments_[GetObstaclePropName("exp_geo_obst_", i) + "_" + link_name_leaf_str] =
          fetch_param("exp_geo_obst_leaf");
      arguments_[GetObstaclePropName("k_geo_obst_", i) + "_" + link_name_leaf_str] =
          fetch_param("k_geo_obst_leaf");
      arguments_[GetObstaclePropName("exp_fin_obst_", i) + "_" + link_name_leaf_str] =
          fetch_param("exp_fin_obst_leaf");
      arguments_[GetObstaclePropName("k_fin_obst_", i) + "_" + link_name_leaf_str] =
          fetch_param("k_fin_obst_leaf");
    }  // End obstacles

    // Plane constraints
    for (auto i = 0; i < task_->GetPlaneConstraintsNum(); ++i) {
      arguments_[fconstraint_prop_name("k_plane_geo_", link_name, i)] = fetch_param("k_plane_geo");
      arguments_[fconstraint_prop_name("exp_plane_geo_", link_name, i)] = fetch_param("exp_plane_geo");
      arguments_[fconstraint_prop_name("k_plane_fin_", link_name, i)] = fetch_param("k_plane_fin");
      arguments_[fconstraint_prop_name("exp_plane_fin_", link_name, i)] = fetch_param("exp_plane_fin");
    }  // End plane constraints
  }    // End collision link names

  // Sundries
  arguments_["base_inertia"] = fetch_param("base_inertia");

  for (auto j = 0; j < robot_->dof(); ++j) {
    const auto j_str = std::to_string(j);
    for (auto i = 0; i < 2; ++i) {
      const auto i_str = std::to_string(i);
      const auto j_i_leaf_str = j_str + "_" + i_str + "_leaf";
      arguments_["exp_limit_fin_limit_joint_" + j_i_leaf_str] = fetch_param("exp_fin_limit_leaf");
      arguments_["exp_limit_geo_limit_joint_" + j_i_leaf_str] = fetch_param("exp_geo_limit_leaf");
      arguments_["k_limit_fin_limit_joint_" + j_i_leaf_str] = fetch_param("k_fin_limit_leaf");
      arguments_["k_limit_geo_limit_joint_" + j_i_leaf_str] = fetch_param("k_geo_limit_leaf");
    }
  }

  for (const auto& [link_name_key, links_pair] : task_->GetSelfCollisionNamePairs()) {
    for (const auto& paired_link_name : links_pair) {
      // NOTE: This concatenation order must match one defined in [FabPlanner::set_components()] as params
      // to add_spherical_self_collision_geometry(), further at [FabSelfCollisionLeaf ctor]
      const auto affix = paired_link_name + "_" + link_name_key;
      arguments_["exp_self_fin_self_collision_" + affix] = fetch_param("exp_fin_self_leaf");
      arguments_["exp_self_geo_self_collision_" + affix] = fetch_param("exp_geo_self_leaf");
      arguments_["k_self_fin_self_collision_" + affix] = fetch_param("k_fin_self_leaf");
      arguments_["k_self_geo_self_collision_" + affix] = fetch_param("k_geo_self_leaf");
    }
  }

  // Attractor args
  for (auto i = 0; i < task_->GetSubGoals().size(); ++i) {
    const auto i_leaf_str = "_goal_" + std::to_string(i) + "_leaf";
    arguments_["attractor_alpha" + i_leaf_str] = fetch_param("attractor_alpha");
    arguments_["attractor_weight" + i_leaf_str] = fetch_param("attractor_weight");
    arguments_["attractor_metric_alpha" + i_leaf_str] = fetch_param("attractor_metric_alpha");
    arguments_["attractor_metric_beta" + i_leaf_str] = fetch_param("attractor_metric_beta");
    arguments_["attractor_metric_scale" + i_leaf_str] = fetch_param("attractor_metric_scale");
  }

  // Damper args
  arguments_["alpha_beta_damper"] = fetch_param("alpha_beta_damper");
  arguments_["beta_close_damper"] = fetch_param("beta_close_damper");
  arguments_["beta_distant_damper"] = fetch_param("beta_distant_damper");
  arguments_["radius_shift_damper"] = fetch_param("radius_shift_damper");
  arguments_["alpha_eta_damper"] = fetch_param("alpha_eta_damper");
  arguments_["alpha_shift_damper"] = fetch_param("alpha_shift_damper");
  arguments_["ex_factor_damper"] = fetch_param("ex_factor_damper");

#if FAB_VERIFY_TUNED_PARAMS
  for (const auto& [param_name, _] : FabParamTuner::get_default_parameters()) {
    if (false == fetched_params_names.contains(param_name)) {
      FAB_PRINT(param_name);
      assert(false);
    }
  }
#endif
  fab_core::print_named_mapdb(arguments_, "TUNED ARG VALS");
}

void FabPlanner::Plan(const FabParamDict& params) {
  const auto robot_dof = robot_->dof();
  std::vector<double> q = task_->QueryJointPos(robot_dof);
  std::vector<double> qdot = task_->QueryJointVel(robot_dof);
  FAB_PRINTDB("QPOS", q);
  FAB_PRINTDB("QVEL", qdot);
  arguments_ = {{"q", std::move(q)}, {"qdot", std::move(qdot)}};

  // [X-space goals] & [Weights of goals]
  SetGoalArguments();

  // [Plane constraints]
  SetConstraintArguments();

  // [Radius collision bodies]
  SetCollisionArguments();

  // [Obstacles' size & pos, vel, etc.]
  SetObstacleArguments();

  /// [Scalar tuned coeffecients]
  if (tuning_on_) {
    SetTuningArguments(params);
  }

  // Compute action
  SetAction(compute_action(arguments_));
}

void FabPlanner::OptimizePolicy(int horizon, mjpc::ThreadPool& pool) {
  // get nominal trajectory
  this->NominalTrajectory(horizon, pool);

  const FabSharedMutexLock lock(policy_mutex_);

  // Tune
  // NOTE: It takes more priority over [Plan], which could proceed as [Tune] finishes
  static bool last_tuning_on = false;
  bool tuning_off_then_on = (!last_tuning_on && tuning_on_);
  last_tuning_on = tuning_on_;
  if (tuning_on_ && !param_tuner_->tuning_finished()) {
    // Fetch last trial tuning result
    param_tuner_->fetch_tune_results();

    // Start next trial tuning
    param_tuner_->tune(tuning_off_then_on);
  }
  // Plan with best params from the tuning
  else {
    Plan(tuning_on_ ? FabParamTuner::get_best_parameters() : FabParamTuner::get_default_parameters());
  }
}