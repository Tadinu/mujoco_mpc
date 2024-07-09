#pragma once

#include <memory>
#include <random>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_core_util.h"

struct FabSubGoal {
  FabSubGoal() = default;
  explicit FabSubGoal(std::string in_name) : name(std::move(in_name)) {}
  std::string name;
  bool is_primary_goal = false;
  double epsilon = 0.;
  std::vector<casadi_int> indices;
  int dimension() const { return indices.size(); }
  double weight = 0.;
  std::string type = "StaticSubGoal";
  std::vector<double> default_values(const double default_val) const {
    std::vector<double> defaults;
    defaults.resize(dimension(), default_val);
    return defaults;
  }
  std::vector<double> desired_position = default_values(0.);
  std::vector<double> desired_vel = default_values(0.);
  std::vector<double> desired_acc = default_values(0.);

  virtual void verify() const {
    const auto pos_size = desired_position.size();
    const auto vel_size = desired_vel.size();
    const auto acc_size = desired_acc.size();
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

  void reset() { desired_position = desired_vel = desired_acc = default_values(0.); }
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

struct FabStaticSubGoal : public FabSubGoal {
  FabStaticSubGoal() = default;
  explicit FabStaticSubGoal(std::string _name) : FabSubGoal(_name) {}
  std::string parent_link_name;
  std::string child_link_name;
  std::vector<double> lower_pos;
  std::vector<double> upper_pos;

  std::vector<double> limit_low_pos() const { return lower_pos.empty() ? default_values(-1.) : lower_pos; }
  std::vector<double> limit_high_pos() const { return upper_pos.empty() ? default_values(1.) : upper_pos; }

  void shuffle_pos() override {
    FabSubGoal::shuffle_pos();
    const auto limit_lows = limit_low_pos();
    const auto limit_highs = limit_high_pos();
    const auto low_limit_size = limit_lows.size();
    for (auto i = 0; i < low_limit_size; ++i) {
      desired_position[i] = rand_val(limit_lows[i], limit_highs[i]);
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

struct FabDynamicGoal : public FabSubGoal {
  // {"spline", "analytic"}
  std::string traj_type;
};

struct FabStaticJointSpaceSubGoal : public FabStaticSubGoal {
  // {"spline", "analytic"}
  std::string traj_type;
};

using FabSubGoalPtr = std::shared_ptr<FabSubGoal>;
using FabSubGoalPtrArray = std::vector<FabSubGoalPtr>;
class FabGoalComposition {
 public:
  static FabSubGoalPtr create_sub_goal(const std::string& sub_goal_type, const std::string& sub_goal_name) {
    if (sub_goal_type == "staticSubGoal") {
      return std::make_shared<FabStaticSubGoal>(sub_goal_name);
    } else if (fab_core::has_collection_element(std::vector<std::string>{"analyticSubGoal", "splineSubGoal"},
                                                sub_goal_type)) {
      return std::make_shared<FabStaticSubGoal>(sub_goal_name);
    } else if (sub_goal_type == "staticJointSpaceSubGoal") {
      return std::make_shared<FabStaticSubGoal>(sub_goal_name);
    }
    throw FabError(sub_goal_type + ": unknown sub-goal type!");
  }

  FabSubGoalPtrArray sub_goals() const { return sub_goals_; }

  FabSubGoalPtr primary_goal() const { return get_goal_by_index(primary_goal_index_); }

  FabSubGoalPtr get_goal_by_name(const std::string& name) const {
    for (const auto& sub_goal : sub_goals_) {
      if (sub_goal->name == name) {
        return sub_goal;
      }
    }
    return nullptr;
  }

  FabSubGoalPtr get_goal_by_index(const int index) const {
    return ((index >= 0) && (index < sub_goals_.size())) ? sub_goals_[index] : nullptr;
  }

  void shuffle_pos() {
    for (auto& sub_goal : sub_goals_) {
      sub_goal->shuffle_pos();
    }
  }

 protected:
  int primary_goal_index_ = -1;
  FabSubGoalPtrArray sub_goals_;
};
