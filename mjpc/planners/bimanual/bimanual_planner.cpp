/**
Panda Bimanual Setup Example with Landing Controller

\author Riddhiman Laha, Juraj Vrabel, Jonathan Vorndamme, and Luis Figueredo
\since 02/2021

**/
#include <stdlib.h>

#include <Eigen/Core>
#include <exception>
#include <fstream>

// mjpc
#include "mjpc/planners/bimanual/bimanual_planner.h"
#include "mjpc/planners/bimanual/controller_type.h"
#include "mjpc/planners/bimanual/end_condition.h"
#include "mjpc/planners/bimanual/goal_type.h"
#include "mjpc/test/lqr.h"

using Eigen::Vector3d;
const int MAX_PATH_LENGTH = 2000;

namespace mjpc {
/***************************************************************
 *     Constructor For PandaBimanual Control Class
 ***************************************************************/
double dead_zone(const double& in, const double& dz) { return std::min(in + dz, std::max(in - dz, 0.0)); }

template <int i, int j>
Eigen::Matrix<double, i, j> dead_zone(const Eigen::Matrix<double, i, j>& in, const double& dz) {
  Eigen::Matrix<double, i, j> tmp;
  for (int k = 0; k < i; ++k) {
    for (int l = 0; l < j; ++l) {
      tmp(k, l) = dead_zone(in(k, l), dz);
    }
  }
  return tmp;
}

void PandaBimanualPlanner::PlanCallback(const Vector3d& p) {
  if (IsPlanningActive()) {
    if (!open_loop_) {
      cf_manager_.setRealEEAgentPosition(p);
    }
    cf_manager_.stopPrediction();
    const int best_agent_id =
        cf_manager_.evaluateAgents(obstacles_, k_goal_dist_, k_path_len_,
                                   k_safe_dist_, k_workspace_, des_ws_limits_);
    if (visualize_predicted_paths_) {
      const auto predicted_paths = cf_manager_.getPredictedPaths();
      for (const auto& predicted_path_i : predicted_paths) {
        if (predicted_path_i.size() > 2) {
          predicted_best_id_ = best_agent_id;
          predicted_paths_.push_back(predicted_path_i);
        }
      }
    }
    cf_manager_.moveRealEEAgent(obstacles_, time_step_, 1, best_agent_id);
    cf_manager_.resetEEAgents(cf_manager_.getNextPosition(),
                              cf_manager_.getNextVelocity(), obstacles_);
    cf_manager_.startPrediction();

    // Move to predicted position
    controller_->targetPoseCallback(cf_manager_.getNextPosition());
    current_dist_from_goal_ = cf_manager_.getDistFromGoal();

    if (visualize_commanded_path_) {
      commanded_path_ = cf_manager_.getPlannedTrajectory();
    }
  } else {
    print("Planning not active. Setting initial position:", p);
    cf_manager_.setInitialPosition(p);
    got_initial_pos_ = true;
  }
}

void PandaBimanualPlanner::InitCFManager() {
  const auto& goal_pos = task_->GetSubGoals()[current_goal_id_]->cfg_.desired_state.pose.pos;
  last_goal_ = Eigen::Map<const Vector3d>(goal_pos.data());
  cf_manager_.init(last_goal_, time_step_, obstacles_, std::vector(num_agents_ee_, k_attr_),
                   std::vector(num_agents_ee_, k_circ_), std::vector(num_agents_ee_, k_repel_),
                   std::vector(num_agents_ee_, k_damp_), std::vector(num_agents_ee_, k_manip_),
                   std::vector(num_agents_body_, k_repel_body_), velocity_, approach_dist_, detect_shell_rad_,
                   max_prediction_steps_, prediction_freq_multiple_);
}

void PandaBimanualPlanner::TaskCallback(GoalType goal_type) {
  const auto subgoals = task_->GetSubGoals();
  if ((current_goal_id_ < 0) || (current_goal_id_ >= subgoals.size())) {
    return;
  }
  goal_type_ = goal_type;
  switch (goal_type) {
    case GoalType::GESTURE: {
      while (WaitForGesture(gesture_th_)) {
      }
      break;
    }
    case GoalType::PLAN: {
      // NOTE: ONLY IF PLANNING IS NON-ACTIVIVE, INITIAL_POS IS RECEIVED
      if (!got_initial_pos_) {
        return;
      }
      Vector3d current_pos = cf_manager_.getNextPosition();
      InitCFManager(); // Initting [cf_manager_.getGoalPosition()] here-in
      cf_manager_.setInitialPosition(current_pos);
      last_pos_ = Vector3d::Zero();
      print("[Currently Non-active]: Planning towards goal:\n", cf_manager_.getGoalPosition().transpose());

      // Plan on [current_pos]
      current_pos.z() += 0.00001;
      // Publish -> "goals": [current_pos]
      controller_->targetPoseCallback(current_pos);
    }
    break;
    case GoalType::GOTO: {
      got_initial_pos_ = false;
      controller_->jointMotionCallback(1., q_goal_);
      break;
    }
    case GoalType::GRASP: {
      CloseGrippers();
      break;
    }
    case GoalType::RELEASE: {
      OpenGrippers(gripper_width_);
      break;
    }
    case GoalType::ERROR_RECOVERY: {
      ErrorRecovery();
      break;
    }
    default: {
      print("Invalid type: ", goal_type.to_string(), "for goal ", current_goal_id_);
      return;
    }
  }
  switch (ec_) {
    case EndCondition::NONE: {
      goal_finished_ = true;
      break;
    }
    case EndCondition::CONTACT: {
      bool contacted =
#if 1
          false;
#else
      ((th_max_ - contact_wrench_).minCoeff() < 0 ||
       (contact_wrench_ - th_min_).minCoeff() < 0);
#endif
      if (contacted) {
        goal_finished_ = true;
      }
      break;
    }
    case EndCondition::REACHED: {
      if (cf_manager_.getDistFromGoal() < 0.01) {
        goal_finished_ = true;
      }
      break;
    }
    case EndCondition::ENDLESS: {
      goal_finished_ = false;
      break;
    }
    default: {
      print("Invalid type: %s for end_condition %i.", end_condition_.c_str(), current_goal_id_);
      return;
    }
  }
  if (goal_finished_) {
    ++current_goal_id_;
    goal_type_ = GoalType::INVALID;
  }
}

/***************************************************************
 *     Method set contact detection thresholds
 ***************************************************************/
void PandaBimanualPlanner::SetThresholds(const Vector6d& F_min,
                                         const Vector6d& F_max) {
  th_min_ = F_min;
  th_max_ = F_max;
}

/***************************************************************
 *     Method to have the robot not holding anything to wait
 *     until it is pushed
 ***************************************************************/
bool PandaBimanualPlanner::WaitForGesture(double F_th) {
  return false;
  //return (contact_wrench_.head<3>().norm() < F_th);
}
} // namespace mjpc
