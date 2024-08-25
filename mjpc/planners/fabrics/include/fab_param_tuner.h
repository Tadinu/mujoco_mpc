#pragma once
#include <casadi/casadi.hpp>
#include <random>

#include "mjpc/optuna/optuna.h"
#include "mjpc/planners/fabrics/include/fab_config.h"
#include "mjpc/planners/fabrics/include/fab_math_util.h"
#include "mjpc/planners/fabrics/include/fab_planner.h"
#include "mjpc/planners/fabrics/include/fab_robot.h"
#include "mjpc/task.h"

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
    const auto robot_dof = robot_->dof();
    q0_ = task_->QueryJointPos(robot_dof);
    qdot_ = task_->QueryJointVel(robot_dof);
    initial_distance_to_goal_0_ = planner_->GetDistanceToGoal(q0_);
    initial_distance_to_obstacles_ = evaluate_distance_to_closest_obstacle();
  }

  static FabParamDict get_default_parameters() {
    static const FabParamDict default_params = {// Base energy
                                                {"base_inertia", 0.20},

                                                // Obstacles
                                                {"exp_geo_obst_leaf", 5},
                                                {"k_geo_obst_leaf", 0.5},
                                                {"exp_fin_obst_leaf", 1},
                                                {"k_fin_obst_leaf", 0.1},
                                                {"exp_geo_self_leaf", 1},
                                                {"k_geo_self_leaf", 0.5},
                                                {"exp_fin_self_leaf", 1},
                                                {"k_fin_self_leaf", -0.1},
                                                {"exp_geo_limit_leaf", 1},
                                                {"k_geo_limit_leaf", 0.1},
                                                {"exp_fin_limit_leaf", 1},
                                                {"k_fin_limit_leaf", 0.1},

                                                // Plane constraint
                                                {"k_plane_geo", 0.5},
                                                {"exp_plane_geo", 5},
                                                {"k_plane_fin", 0.1},
                                                {"exp_plane_fin", 1},

                                                // Attractor
                                                {"attractor_alpha", 10},
                                                {"attractor_weight", 5},
                                                {"attractor_metric_alpha", 2},
                                                {"attractor_metric_beta", 0.3},
                                                {"attractor_metric_scale", 0.3},

                                                // Damper
                                                {"alpha_beta_damper", 0.5},
                                                {"beta_distant_damper", 0.01},
                                                {"beta_close_damper", 6.5},
                                                {"radius_shift_damper", 0.05},
                                                {"alpha_eta_damper", 0.9},
                                                {"alpha_shift_damper", 0.5},
                                                {"ex_factor_damper", 0.5}};
    return default_params;
  }

  static FabParamDict get_best_parameters() { return best_params_; }

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
        // Base energy
        {"base_inertia", {{"low", 0.01}, {"high", 1.0}, {"int", false}, {"log", false}}},

        // Obstacles
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

        // Plane constraint
        {"k_plane_geo", {{"low", 0.01}, {"high", 1}, {"int", false}, {"log", true}}},
        {"exp_plane_geo", {{"low", 1}, {"high", 5}, {"int", true}, {"log", true}}},
        {"k_plane_fin", {{"low", 0.01}, {"high", 1}, {"int", false}, {"log", true}}},
        {"exp_plane_fin", {{"low", 1}, {"high", 5}, {"int", true}, {"log", true}}},

        // Attractor
        {"attractor_alpha", {{"low", 0.0}, {"high", 50}, {"int", false}, {"log", false}}},
        {"attractor_weight", {{"low", 0.0}, {"high", 100}, {"int", false}, {"log", false}}},
        {"attractor_metric_alpha", {{"low", 1.0}, {"high", 5}, {"int", false}, {"log", false}}},
        {"attractor_metric_beta", {{"low", 0.1}, {"high", 5}, {"int", false}, {"log", false}}},
        {"attractor_metric_scale", {{"low", 0.1}, {"high", 1}, {"int", false}, {"log", false}}},

        // Damper
        {"alpha_beta_damper", {{"low", 0.0}, {"high", 1}, {"int", false}, {"log", false}}},
        {"beta_distant_damper", {{"low", 0.0}, {"high", 1.0}, {"int", false}, {"log", false}}},
        {"beta_close_damper", {{"low", 5.0}, {"high", 20.0}, {"int", false}, {"log", false}}},
        {"radius_shift_damper", {{"low", 0.01}, {"high", 0.1}, {"int", false}, {"log", false}}},
        {"alpha_eta_damper", {{"low", 0.0}, {"high", 1}, {"int", false}, {"log", false}}},
        {"alpha_shift_damper", {{"low", 0.0}, {"high", 1}, {"int", false}, {"log", false}}},
        {"ex_factor_damper", {{"low", 1.0}, {"high", 30.0}, {"int", false}, {"log", false}}}};
    assert(search_space_data_.size() == get_default_parameters().size());
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
    FAB_PRINT("OPTUNA TOTAL COST", sum);
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
    study_->tell(*trials_[current_trial_idx_], total_costs(get_trial_result()));

    if (current_trial_idx_ == (trials_.size() - 1)) {
      const optuna::FrozenTrial best_trial = study_->best_trial();
      FAB_PRINT("BEST TRIAL", best_trial.number, best_trial.value);

      // Refresh [best_params_]
      best_params_.clear();
      for (const auto& [param_name, param_val] : get_default_parameters()) {
        if (best_trial.contains(param_name)) {
          best_params_.insert_or_assign(param_name, best_trial.param_to<double>(param_name));
        }
      }
      // FAB_PRINT("Saving study");
      // save_study();
    }
    current_trial_idx_++;
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
    CaSX distance_to_obstacles = 10000;
    for (const auto& link_name : task_->GetCollisionLinkNames()) {
      const CaSX fk = robot_->fk()->casadi(q, link_name, task_->GetBaseBodyName(),
                                           fab_math::CASX_TRANSF_IDENTITY, true /*position_only*/);

      for (const auto& obst : task_->GetObstacleStatesX()) {
        std::vector obst_pos(3, 0.);
        memcpy(obst_pos.data(), obst.pos_.data(), 3 * sizeof(double));
        distance_to_obstacles = CaSX::fmin(distance_to_obstacles, double(CaSX::norm_2(obst_pos - fk)));
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

  double evaluate_pace_to_goal() const {
    return planner_->GetDistanceToGoal(q0_) / initial_distance_to_goal_0_;
  }

  double evaluate_distance_to_closest_obstacle() const {
    return static_cast<double>(collision_metric_(CaSX(q0_))[0]);
  }

  void eval(const FabParamDict& params) {
    // Plan
    planner_->Plan(params);

    // Compute [path_length_, distances_to_goal_0_, distances_to_closest_obstacle_]
    const auto robot_dof = robot_->dof();
    q0_ = task_->QueryJointPos(robot_dof);
    static auto q_last = q0_;
    qdot_ = task_->QueryJointVel(robot_dof);
    FAB_PRINTDB("QPOS", q0_);
    FAB_PRINTDB("QVEL", qdot_);
    path_length_ += double(CaSX::norm_2(CaSX(q0_) - CaSX(q_last)).scalar());
    q_last = q0_;
    distances_to_goal_0_.push_back(evaluate_pace_to_goal());
    distances_to_closest_obstacle_.push_back(evaluate_distance_to_closest_obstacle());
    FAB_PRINT("current_trial_idx_", current_trial_idx_);
    FAB_PRINT("initial_distance_to_obstacles_", initial_distance_to_obstacles_);
    FAB_PRINT("distances_to_goal_0_", distances_to_goal_0_.back());
    FAB_PRINT("distances_to_closest_obstacle_", distances_to_closest_obstacle_.back());
  }

  FabParamWeightDict get_trial_result() const {
    return {{"path_length_", path_length_ / 10},
            {"pace_to_goal", std::reduce(distances_to_goal_0_.begin(), distances_to_goal_0_.end()) /
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
  static constexpr int TRIALS_NUM = 100;
  std::vector<optuna::TrialPtr> trials_ = std::vector<optuna::TrialPtr>(TRIALS_NUM, nullptr);
  int current_trial_idx_ = 0;
  bool last_trial_started_ = false;
  bool last_trial_result_fetched_ = true;

  std::vector<double> q0_;
  std::vector<double> qdot_;
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
  FabParamWeightDict param_weights_ = {{"path_length_", 0.4}, {"pace_to_goal", 0.4}, {"obstacles", 0.2}};
  static FabParamDict best_params_;
};

FabParamDict FabParamTuner::best_params_ = get_default_parameters();
const std::string FabParamTuner::DEFAULT_STUDY_DB_PATH = "sqlite:///fab_param_study.db";
