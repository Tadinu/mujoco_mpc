#pragma once

#include <memory>
#include <random>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_core_util.h"

enum class FabSubGoalType : uint8_t { STATIC, STATIC_JOINT_SPACE, DYNAMIC /* Analytic, Spline, etc.*/ };

enum class FabSubGoalTrajectoryType : uint8_t { ANALYTIC, SPLINE };

struct FabSubGoalConfig {
  std::string name;
  FabSubGoalType type = FabSubGoalType::STATIC;
  bool is_primary_goal = false;
  double epsilon = 0.;
  std::vector<casadi_int> indices;
  double weight = 0.;
  std::string parent_link_name;
  std::string child_link_name;
  std::vector<double> lower_pos;
  std::vector<double> upper_pos;
  std::vector<double> desired_position = default_values(0.);
  std::vector<double> desired_vel = default_values(0.);
  std::vector<double> desired_acc = default_values(0.);

  FabSubGoalTrajectoryType traj_type = FabSubGoalTrajectoryType::ANALYTIC;

  size_t dimension() const { return indices.size(); }
  std::vector<double> default_values(const double default_val) const {
    std::vector<double> defaults;
    defaults.resize(dimension(), default_val);
    return defaults;
  }

  void clear() { desired_position = desired_vel = desired_acc = default_values(0.); }
};

struct FabSubGoal {
  FabSubGoal() = default;
  explicit FabSubGoal(FabSubGoalConfig config) : cfg_(std::move(config)) {}
  FabSubGoalConfig cfg_;

  std::string name() const { return cfg_.name; }
  size_t dimension() const { return cfg_.dimension(); }
  FabSubGoalType type() const { return cfg_.type; }

  std::vector<casadi_int> indices() const { return cfg_.indices; }
  double weight() const { return cfg_.weight; }
  std::string parent_link_name() const { return cfg_.parent_link_name; }
  std::string child_link_name() const { return cfg_.child_link_name; }

  std::vector<double> limit_low_pos() const {
    return cfg_.lower_pos.empty() ? cfg_.default_values(-1.) : cfg_.lower_pos;
  }
  std::vector<double> limit_high_pos() const {
    return cfg_.upper_pos.empty() ? cfg_.default_values(1.) : cfg_.upper_pos;
  }

  bool is_primary_goal() const { return cfg_.is_primary_goal; }

  virtual void verify() const {
    const auto pos_size = cfg_.desired_position.size();
    const auto vel_size = cfg_.desired_vel.size();
    const auto acc_size = cfg_.desired_acc.size();
    if (pos_size != dimension()) {
      throw FabError(std::string("Desired position size: ") + std::to_string(pos_size) +
                     " does not match dimension one: " + std::to_string(dimension()));
    }
    if (vel_size != pos_size) {
      throw FabError(std::string("Desired velocity size: ") + std::to_string(pos_size) +
                     " does not match position: " + std::to_string(vel_size));
    }
    if (acc_size != pos_size) {
      throw FabError(std::string("Desired position size ") + std::to_string(pos_size) +
                     " does not match dimension: " + std::to_string(acc_size));
    }
  }

  void reset() { cfg_.clear(); }
  virtual void shuffle_pos() {
    verify();
    reset();
  }

  // PRN
  std::random_device rd;
  // Standard mersenne_twister_engine seeded with rd()
  std::mt19937 gen = std::mt19937(rd());
  double rand_val(const double lower, const double upper) {
    std::uniform_real_distribution<> dis = std::uniform_real_distribution<>(lower, upper);
    return dis(gen);
  }
};
using FabSubGoalPtr = std::shared_ptr<FabSubGoal>;

struct FabStaticSubGoal : public FabSubGoal {
  FabStaticSubGoal() = default;
  explicit FabStaticSubGoal(FabSubGoalConfig config) : FabSubGoal(std::move(config)) {}
  void shuffle_pos() override {
    FabSubGoal::shuffle_pos();
    const auto limit_lows = limit_low_pos();
    const auto limit_highs = limit_high_pos();
    const auto low_limit_size = limit_lows.size();
    for (auto i = 0; i < low_limit_size; ++i) {
      cfg_.desired_position[i] = rand_val(limit_lows[i], limit_highs[i]);
    }
  }

  void verify() const override {
    FabSubGoal::verify();
    const auto lower_pos_size = limit_low_pos().size();
    const auto upper_pos_size = limit_high_pos().size();
    if (lower_pos_size != upper_pos_size) {
      throw FabError(std::string("Lower position limits' size: ") + std::to_string(lower_pos_size) +
                     "does not match upper one: " + std::to_string(lower_pos_size));
    }
  }
};

struct FabDynamicSubGoal : public FabSubGoal {
  FabDynamicSubGoal() = default;
  explicit FabDynamicSubGoal(FabSubGoalConfig config) : FabSubGoal(std::move(config)) {}
};

struct FabStaticJointSpaceSubGoal : public FabStaticSubGoal {
  FabStaticJointSpaceSubGoal() = default;
  explicit FabStaticJointSpaceSubGoal(FabSubGoalConfig config) : FabStaticSubGoal(std::move(config)) {}
};

using FabSubGoalPtr = std::shared_ptr<FabSubGoal>;
using FabSubGoalPtrArray = std::vector<FabSubGoalPtr>;

// ===========================================================================================================
// GOAL COMPOSITON --
//
using FabSubGoalConfigItem =
    FabVariant<bool, double, std::string, std::vector<double>, std::vector<casadi_int>>;
using FabGoalConfig = std::map<std::string, std::map<std::string, FabSubGoalConfigItem>>;

class FabGoalComposition {
public:
  template <typename T, typename = std::enable_if<std::is_base_of_v<T, FabSubGoal>>>
  static std::shared_ptr<T> create_sub_goal(FabSubGoalConfig goal_config) {
    return std::make_shared<T>(std::move(goal_config));
  }

  FabGoalComposition() = default;
  FabGoalComposition(std::string name, const FabGoalConfig& config) : name_(std::move(name)) {
    for (const auto& [sub_goal_name, subgoal_config_map] : config) {
      const auto type_text = fab_core::get_variant_value<std::string>(subgoal_config_map.at("type"));
      auto sub_goal = std::make_shared<FabStaticSubGoal>(FabSubGoalConfig{
          .name = sub_goal_name,
          .type = (type_text == "static")               ? FabSubGoalType::STATIC
                  : (type_text == "static_joint_space") ? FabSubGoalType::STATIC_JOINT_SPACE
                                                        : FabSubGoalType::DYNAMIC,
          .is_primary_goal = fab_core::get_variant_value<bool>(subgoal_config_map.at("is_primary_goal")),
          .epsilon = fab_core::get_variant_value<double>(subgoal_config_map.at("epsilon")),
          .indices = fab_core::get_variant_value<std::vector<casadi_int>>(subgoal_config_map.at("indices")),
          .weight = fab_core::get_variant_value<double>(subgoal_config_map.at("weight")),
          .parent_link_name = fab_core::get_variant_value<std::string>(subgoal_config_map.at("parent_link")),
          .child_link_name = fab_core::get_variant_value<std::string>(subgoal_config_map.at("child_link")),
          .desired_position =
              fab_core::get_variant_value<std::vector<double>>(subgoal_config_map.at("desired_pos")),
          .desired_vel =
              fab_core::get_variant_value<std::vector<double>>(subgoal_config_map.at("desired_vel")),
          .desired_acc =
              fab_core::get_variant_value<std::vector<double>>(subgoal_config_map.at("desired_acc"))});
      add_sub_goal(sub_goal);
    }
  }

  FabSubGoalPtrArray sub_goals() const { return sub_goals_; }

  FabSubGoalPtr primary_goal() const { return get_goal_by_index(primary_goal_index_); }

  FabSubGoalPtr get_goal_by_name(const std::string& name) const {
    for (const auto& sub_goal : sub_goals_) {
      if (sub_goal->name() == name) {
        return sub_goal;
      }
    }
    return nullptr;
  }

  FabSubGoalPtr get_goal_by_index(const int index) const {
    return ((index >= 0) && (index < sub_goals_.size())) ? sub_goals_[index] : nullptr;
  }

  void add_sub_goal(FabSubGoalPtr sub_goal) { sub_goals_.emplace_back(std::move(sub_goal)); }

  void shuffle_pos() {
    for (auto& sub_goal : sub_goals_) {
      sub_goal->shuffle_pos();
    }
  }

  bool is_valid() const { return !sub_goals_.empty(); }

protected:
  std::string name_;
  int primary_goal_index_ = -1;
  FabSubGoalPtrArray sub_goals_;
};
