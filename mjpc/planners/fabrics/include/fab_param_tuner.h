#pragma once
#include <casadi/casadi.hpp>
#include <random>

#include "mjpc/optuna/optuna.h"
#include "mjpc/planners/fabrics/include/fab_config.h"
#include "mjpc/planners/fabrics/include/fab_math_util.h"
#include "mjpc/planners/fabrics/include/fab_planner.h"
#include "mjpc/planners/fabrics/include/fab_robot.h"
#include "mjpc/task.h"

using FabParamWeightDict = std::map<const char*, double>;
using FabSearchSpaceData = std::map<const char*, FabParamWeightDict>;

class FabParamTuner {
public:
  FabParamTuner() { initialize(); }
  virtual ~FabParamTuner() = default;
  explicit FabParamTuner(FabParamWeightDict param_weights) : param_weights_(std::move(param_weights)) {
    initialize();
  }

  void initialize() {
    init_study();
    init_search_space();
  }

  void set_planner(FabPlanner* planner) {
    planner_ = planner;
    robot_ = planner->robot();
    task_ = planner->task();
#if 0
    // NOTE: Env info (obstacles, etc.) is fetched directly through [task_]
    env_ = FabEnvironment(task_->GetTotalObstaclesNum(), 0, 1);
#endif
    create_collision_metric();
    initial_distance_to_obstacles_ =
        evaluate_distance_to_closest_obstacle(task_->QueryJointPos(robot_->dof()));
  }

  static FabParamWeightDict get_manual_parameters() {
    return {{"exp_geo_obst_leaf", 3},   {"k_geo_obst_leaf", 0.03},
            {"exp_fin_obst_leaf", 3},   {"k_fin_obst_leaf", 0.03},
            {"exp_geo_self_leaf", 3},   {"k_geo_self_leaf", 0.03},
            {"exp_fin_self_leaf", 3},   {"k_fin_self_leaf", 0.03},
            {"exp_geo_limit_leaf", 2},  {"k_geo_limit_leaf", 0.3},
            {"exp_fin_limit_leaf", 3},  {"k_fin_limit_leaf", 0.05},
            {"weight_attractor", 2},    {"base_inertia", 0.20},
            {"alpha_b_damper", 0.5},    {"beta_distant_damper", 0.01},
            {"beta_close_damper", 6.5}, {"radius_shift_damper", 0.05},
            {"ex_factor", 15.0}};
  }

  static FabParamWeightDict get_caspar_parameters() {
    return {
        {"exp_geo_obst_leaf", 2},        //[1, 5]
        {"exp_geo_self_leaf", 2},        //[1, 5]
        {"exp_geo_limit_leaf", 2},       //[1, 5]
        {"exp_fin_obst_leaf", 2},        //[1, 5]
        {"exp_fin_self_leaf", 2},        //[1, 1]
        {"exp_fin_limit_leaf", 2},       //[1, 5]
        {"k_geo_obst_leaf", 0.01},       //[0.01, 1]
        {"k_geo_self_leaf", 0.01},       //[0.01, 1]
        {"k_geo_limit_leaf", 0.01},      //[0.01, 1]
        {"k_fin_self_leaf", 0.01},       //[0.01, 1]
        {"k_fin_obst_leaf", 0.01},       //[0.01, 1]
        {"k_fin_limit_leaf", 0.01},      //[0.01, 1]
        {"base_inertia", 0.50},          //[0, 1]
        {"alpha_b_damper", 0.7},         //[0, 1]
        {"beta_distant_damper", 0.},     //[0, 1]
        {"beta_close_damper", 9},        //[5, 20]
        {"radius_shift_damper", 0.050},  //[0.01, 0.1]
        {"ex_factor", 30.0}              //[1, 30]
    };
  }

  static const std::string DEFAULT_STUDY_DB_PATH;
  void init_study(const std::string& study_name = "fab_param_tuner_study",
                  const std::string& study_file_path = DEFAULT_STUDY_DB_PATH) {
    bool storage_exists = !study_file_path.empty() &&
#if 1
                          bool(std::ifstream(study_file_path.c_str()));
#else
                          std::filesystem::exists(study_file_path);
#endif
    if (storage_exists) {
      FAB_PRINT("[FabParamTuner] Reading study from " + study_file_path);
      // TODO: tbd
    } else {
      FAB_PRINT("[FabParamTuner] Create new study with backend:",
                study_file_path.empty() ? "in-memory" : study_file_path);
      study_ = std::make_shared<optuna::Study>(study_name, study_file_path, optuna::MINIMIZE, true);
    }
  }

  void init_search_space() {
    search_space_data_ = {
        {"exp_geo_obst_leaf", {{"low", 1}, {"high", 5}, {"int", true}, {"log", false}}},
        {"exp_geo_self_leaf", {{"low", 1}, {"high", 5}, {"int", true}, {"log", false}}},
        {"exp_geo_limit_leaf", {{"low", 1}, {"high", 5}, {"int", true}, {"log", false}}},
        {"exp_fin_obst_leaf", {{"low", 1}, {"high", 5}, {"int", true}, {"log", false}}},
        {"exp_fin_self_leaf", {{"low", 1}, {"high", 5}, {"int", true}, {"log", false}}},
        {"exp_fin_limit_leaf", {{"low", 1}, {"high", 5}, {"int", true}, {"log", false}}},
        {"k_geo_obst_leaf", {{"low", 0.01}, {"high", 1}, {"int", false}, {"log", true}}},
        {"k_geo_self_leaf", {{"low", 0.01}, {"high", 1}, {"int", false}, {"log", true}}},
        {"k_geo_limit_leaf", {{"low", 0.01}, {"high", 1}, {"int", false}, {"log", true}}},
        {"k_fin_obst_leaf", {{"low", 0.01}, {"high", 1}, {"int", false}, {"log", true}}},
        {"k_fin_self_leaf", {{"low", 0.01}, {"high", 1}, {"int", false}, {"log", true}}},
        {"k_fin_limit_leaf", {{"low", 0.01}, {"high", 1}, {"int", false}, {"log", true}}},
        {"alpha_b_damper", {{"low", 0.0}, {"high", 1}, {"int", false}, {"log", false}}},
        {"base_inertia", {{"low", 0.01}, {"high", 1.0}, {"int", false}, {"log", false}}},
        {"beta_distant_damper", {{"low", 0.0}, {"high", 1.0}, {"int", false}, {"log", false}}},
        {"beta_close_damper", {{"low", 5.0}, {"high", 20.0}, {"int", false}, {"log", false}}},
        {"radius_shift_damper", {{"low", 0.01}, {"high", 0.1}, {"int", false}, {"log", false}}},
        {"ex_factor", {{"low", 1.0}, {"high", 30.0}, {"int", false}, {"log", false}}}};
    for (const auto& [name, data] : search_space_data_) {
      if (data.at("int") > 0) {
        search_space_.add_int(name, static_cast<int>(data.at("low")), static_cast<int>(data.at("high")),
                              1 /*step*/, static_cast<int>(data.at("log")));
      } else {
        search_space_.add_float(name, data.at("low"), data.at("high"), 0 /*step*/,
                                static_cast<bool>(data.at("log")));
      }
    }
  }

  double total_costs(const FabParamWeightDict& costs) const {
    double sum = 0;
    for (const auto& [param_name, param_weight] : param_weights_) {
      sum += param_weight * costs.at(param_name);
    }
    return sum;
  }

  FabParamWeightDict get_random_parameters() const {
    FabParamWeightDict parameters;
    for (const auto& [param_name, param_space] : search_space_data_) {
      if (param_space.at("int") > 0) {
        parameters[param_name] = FabRandom::rand<int>(static_cast<int>(param_space.at("low")),
                                                      static_cast<int>(param_space.at("high")));
      } else {
        parameters[param_name] =
            param_space.at("low") + FabRandom::rand() * (param_space.at("high") - param_space.at("low"));
      }
    }
    return parameters;
  }

  void set_goal_arguments() {
    // [X-space goals] & [Weights of goals]
    const auto sub_goals = task_->GetSubGoals();
    for (auto i = 0; i < sub_goals.size(); ++i) {
      const auto& sub_goal = sub_goals[i];
      const auto i_str = std::to_string(i);
      arguments_.insert_or_assign("x_goal_" + i_str, sub_goal->cfg_.desired_position);
      arguments_.insert_or_assign("weight_goal_" + i_str, sub_goal->cfg_.weight);
    }

#if 0
    // [Plane constraints]
    for (auto i = 0; i < task_->GetPlaneConstraintsNum(); ++i) {
      arguments_.insert_or_assign("constraint_" + std::to_string(i), std::vector<double>{0, 0, 1, 0.0});
    }
#endif
  }

  void set_collision_arguments() {
    const auto collision_link_props = task_->GetCollisionLinkProps();
    for (const auto& link_name : task_->GetCollisionLinkNames()) {
      arguments_["radius_body_" + link_name] = collision_link_props.at(link_name);
    }
  }

  void set_other_arguments(const FabParamWeightDict& params) {
    const auto obstacle_statesX = task_->GetObstacleStatesX();
    const bool fixed_obstacles = task_->AreObstaclesFixed();

    const auto fobstacle_prop_name = [&fixed_obstacles](const char* prefix, const int i) {
      return (fixed_obstacles ? prefix : (std::string(prefix) + "dynamic_")) + std::to_string(i);
    };
    const int dim = task_->GetObstaclesDim();
    std::vector obst_pos(dim, 0.);
    std::vector obst_vel(dim, 0.);
    std::vector obst_acc(dim, 0.);
    for (const auto& link_name : task_->GetCollisionLinkNames()) {
      for (auto i = 0; i < obstacle_statesX.size(); ++i) {
        const auto& obstacle_i = obstacle_statesX[i];
        arguments_.insert_or_assign(fobstacle_prop_name("radius_obst_", i),
                                    FAB_OBSTACLE_SIZE_SCALE * obstacle_i.size_[0]);

        // [obst_pos]
        memcpy(obst_pos.data(), obstacle_i.pos_.data(), dim * sizeof(double));
        arguments_.insert_or_assign(fobstacle_prop_name("x_obst_", i), obst_pos);

        if (!fixed_obstacles) {
          // [obst_vel]
          memcpy(obst_vel.data(), obstacle_i.vel_.data(), dim * sizeof(double));
          arguments_.insert_or_assign(fobstacle_prop_name("xdot_obst_", i), obst_vel);

          // [obst_acc]
          memcpy(obst_acc.data(), obstacle_i.acc_.data(), dim * sizeof(double));
          arguments_.insert_or_assign(fobstacle_prop_name("xddot_obst_", i), obst_acc);
        }

        // LINK OBSTACLE LEAF
        const auto link_name_leaf_str = link_name + "_leaf";
        arguments_.insert_or_assign(fobstacle_prop_name("exp_geo_obst_", i) + "_" + link_name_leaf_str,
                                    params.at("exp_geo_obst_leaf"));
        arguments_.insert_or_assign(fobstacle_prop_name("k_geo_obst_", i) + "_" + link_name_leaf_str,
                                    params.at("k_geo_obst_leaf"));
        arguments_.insert_or_assign(fobstacle_prop_name("exp_fin_obst_", i) + "_" + link_name_leaf_str,
                                    params.at("exp_fin_obst_leaf"));
        arguments_.insert_or_assign(fobstacle_prop_name("k_fin_obst_", i) + "_" + link_name_leaf_str,
                                    params.at("k_fin_obst_leaf"));
      }
    }
    for (auto j = 0; j < robot_->dof(); ++j) {
      const auto j_str = std::to_string(j);
      for (auto i = 0; i < 2; ++i) {
        const auto i_str = std::to_string(i);
        const auto j_i_leaf_str = j_str + "_" + i_str + "_leaf";
        arguments_["exp_limit_fin_limit_joint_" + j_i_leaf_str] = params.at("exp_fin_limit_leaf");
        arguments_["exp_limit_geo_limit_joint_" + j_i_leaf_str] = params.at("exp_geo_limit_leaf");
        arguments_["k_limit_fin_limit_joint_" + j_i_leaf_str] = params.at("k_fin_limit_leaf");
        arguments_["k_limit_geo_limit_joint_" + j_i_leaf_str] = params.at("k_geo_limit_leaf");
      }
    }

    for (const auto& [link_name_key, links_pair] : task_->GetSelfCollisionNamePairs()) {
      for (const auto& paired_link_name : links_pair) {
        // NOTE: This concatenation order must match one defined in [FabPlanner::set_components()] as params
        // to add_spherical_self_collision_geometry(), further at [FabSelfCollisionLeaf ctor]
        const auto affix = paired_link_name + "_" + link_name_key;
        arguments_["exp_self_fin_self_collision_" + affix] = params.at("exp_fin_self_leaf");
        arguments_["exp_self_geo_self_collision_" + affix] = params.at("exp_geo_self_leaf");
        arguments_["k_self_fin_self_collision_" + affix] = params.at("k_fin_self_leaf");
        arguments_["k_self_geo_self_collision_" + affix] = params.at("k_geo_self_leaf");
      }
    }

    // damper arguments
    arguments_["alpha_b_damper"] = params.at("alpha_b_damper");
    arguments_["beta_close_damper"] = params.at("beta_close_damper");
    arguments_["beta_distant_damper"] = params.at("beta_distant_damper");
    arguments_["radius_shift_damper"] = params.at("radius_shift_damper");
    arguments_["base_inertia"] = params.at("base_inertia");
    arguments_["ex_factor_damper"] = params.at("ex_factor");
  }

  virtual void shuffle_env() {}

  void tune(bool restart_tuning = false) {
    if (restart_tuning) {
      // Ditch whatever last trials & reset afresh
      current_trial_idx_ = 0;
    } else {
      if (false == (is_valid_trial_idx(current_trial_idx_) && last_trial_result_fetched_)) {
        return;
      }
    }

    // Always reset accumulative data before running trial regardless
    reset();

    // NOTE: This only triggers a trial run, which would span over multiple frames computing action,
    // accumulating results
    FAB_PRINT("Start running trial id ", current_trial_idx_);
    run_trial(trials_[current_trial_idx_]);
    last_trial_started_ = true;
  }

  void fetch_tune_results() {
    if (false == (is_valid_trial_idx(current_trial_idx_) && last_trial_started_)) {
      return;
    }
    study_->tell(*trials_[current_trial_idx_], total_costs(get_tune_result()));

    if (current_trial_idx_ == (trials_.size() - 1)) {
      FAB_PRINT("Best trial");
      const optuna::FrozenTrial best_trial = study_->best_trial();
      FAB_PRINT(best_trial.number, best_trial.value);
      // FAB_PRINT("Saving study");
      // save_study();
    } else {
      current_trial_idx_++;
    }
    last_trial_result_fetched_ = true;
  }

  bool last_trial_started() const { return last_trial_started_; }
  bool last_trial_finished() const { return last_trial_result_fetched_; }
  bool tuning_finished() const { return current_trial_idx_ >= trials_.size(); }

protected:
  void run_trial(std::shared_ptr<optuna::Trial>& trial) {
    // ob = env.reset(pos=q0)
    // env, obstacles, goal = self.shuffle_env(env,shuffle =shuffle)
    if (false == task_->AreObstaclesFixed()) {
      create_collision_metric();
    }
    eval(sample_fabrics_params_uniform(trial));
  }

  void create_collision_metric() {
    const CaSX q = CaSX::sym("q", robot_->dof());
    double distance_to_obstacles = 10000;
    for (const auto& link_name : task_->GetCollisionLinkNames()) {
      const CaSX fk = robot_->fk()->casadi(q, link_name, task_->GetBaseBodyName(),
                                           fab_math::CASX_TRANSF_IDENTITY, true /*position_only*/);

      for (const auto& obst : task_->GetObstacleStatesX()) {
        std::vector obst_pos(3, 0.);
        memcpy(obst_pos.data(), obst.pos_.data(), 3 * sizeof(double));
        distance_to_obstacles =
            std::fmin(distance_to_obstacles, double(CaSX::norm_2(obst_pos - fk).scalar()));
      }
    }
    collision_metric_ = CaFunction("collision_metric", {q}, {distance_to_obstacles});
  }

  FabParamWeightDict sample_fabrics_params_uniform(std::shared_ptr<optuna::Trial>& trial) {
    trial = std::make_shared<optuna::Trial>(study_->ask(search_space_));
    FabParamWeightDict parameters;
    for (const auto& [param_name, param_space] : search_space_data_) {
      if (param_space.at("int") > 0) {
        parameters[param_name] = trial->param<int>(param_name);
      } else {
        parameters[param_name] = trial->param<float>(param_name);
      }
    }
    return parameters;
  }

  double evaluate_distance_to_goal(const std::vector<double>& q) const {
    const auto& sub_goal_0_cfg = task_->GetSubGoals()[0]->cfg_;
    const auto& sub_goal_0_position = sub_goal_0_cfg.desired_position;
    const CaSX fk = robot_->fk()->casadi(q, sub_goal_0_cfg.child_link_name, task_->GetBaseBodyName(),
                                         fab_math::CASX_TRANSF_IDENTITY, true /*position_only*/);
    return double(CaSX::norm_2(CaSX(sub_goal_0_position) - fk).scalar()) / initial_distance_to_goal_0_;
  }

  double evaluate_distance_to_closest_obstacle(const CaSX& q) const {
    return static_cast<double>(collision_metric_(q)[0]);
  }

  void eval(const FabParamWeightDict& params = get_manual_parameters()) {
    const auto robot_dof = robot_->dof();

    std::vector<double> q0 = task_->QueryJointPos(robot_dof);
    auto q_old = q0;
    std::vector<double> qdot = task_->QueryJointVel(robot_dof);
    FAB_PRINTDB("QPOS", q0);
    FAB_PRINTDB("QVEL", qdot);

#if 1
    // Setup [arguments_]
    arguments_.clear();
    set_goal_arguments();
    set_collision_arguments();
    set_other_arguments(params);

    // [arguments_] -> [args]
    FabCasadiArgMap args = {{"q", q0}, {"qdot", qdot}};
    for (const auto& [arg_name, arg] : arguments_) {
      args.insert_or_assign(arg_name, arg);
    }
    planner_->set_action(planner_->compute_action(args));
#else
    planner_->plan();
#endif
    path_length_ += double(CaSX::norm_2(CaSX(q0) - CaSX(q_old)).scalar());
    q_old = q0;
    distances_to_goal_0_.push_back(evaluate_distance_to_goal(q0));
    distances_to_closest_obstacle_.push_back(evaluate_distance_to_closest_obstacle(q0));
  }

  FabParamWeightDict get_tune_result() const {
    return {{"path_length_", path_length_ / 10},
            {"time_to_goal", std::reduce(distances_to_goal_0_.begin(), distances_to_goal_0_.end()) /
                                 int(distances_to_goal_0_.size())},
            {"obstacles", 1 - *std::min_element(distances_to_closest_obstacle_.begin(),
                                                distances_to_closest_obstacle_.end()) /
                                  initial_distance_to_obstacles_}};
  }

  bool is_valid_trial_idx(int trial_idx) const { return (trial_idx >= 0 && trial_idx < trials_.size()); }

  void reset() {
    last_trial_started_ = false;
    last_trial_result_fetched_ = false;
    path_length_ = 0.0;
    initial_distance_to_goal_0_ = 0.0;
    initial_distance_to_obstacles_ = 0.0;
    distances_to_goal_0_.clear();
    distances_to_closest_obstacle_.clear();
  }

protected:
  FabPlanner* planner_ = nullptr;
  FabRobotPtr robot_ = nullptr;
  mjpc::Task* task_ = nullptr;
#if 0
  // NOTE: Env info (obstacles, etc.) is fetched directly through [task_]
  FabEnvironment env_;
#endif
  FabCasadiArgMap arguments_;
  double dt_ = 0.05;
  static constexpr int TRIALS_NUM = 10;
  std::vector<optuna::TrialPtr> trials_ = std::vector<optuna::TrialPtr>(TRIALS_NUM, nullptr);
  int current_trial_idx_ = 0;
  bool last_trial_started_ = false;
  bool last_trial_result_fetched_ = true;
  double path_length_ = 0.0;
  double initial_distance_to_goal_0_ = 0.0;
  double initial_distance_to_obstacles_ = 0.0;
  std::vector<double> distances_to_goal_0_;
  std::vector<double> distances_to_closest_obstacle_;
  CaFunction collision_metric_;

  // Optuna
  std::shared_ptr<optuna::Study> study_ = nullptr;
  FabSearchSpaceData search_space_data_;
  optuna::SearchSpace search_space_;
  FabParamWeightDict param_weights_ = {{"path_length_", 0.4}, {"time_to_goal", 0.4}, {"obstacles", 0.2}};
};

const std::string FabParamTuner::DEFAULT_STUDY_DB_PATH = "sqlite:///fab_param_study.db";
