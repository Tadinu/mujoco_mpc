// Copyright (c) 2020 Lehrstuhl für Robotik und Sysstemintelligenz, TU München
#include "mjpc/planners/bimanual/dual_panda_costp_controller.h"

#include <cmath>
#include <functional>
#include <memory>
#include <thread>

// mujoco
#include <mujoco/mujoco.h>

// mjpc
#include "mjpc/utilities.h"
#include "mjpc/utils/mjpc_core_util.h"
#include "mjpc/planners/bimanual/joint_motion_generator.h"
#include "mjpc/planners/bimanual/panda_joint_impedance_controller.h"
#include "mjpc/planners/bimanual/bimanual_planner.h"

namespace mjpc {
Vector3d DualPandaCoSTPController::B_T_O_l(Vector3d p_0) { return Vector3d(r_B_O_l_ + B_R_O_l_ * p_0); }

Vector3d DualPandaCoSTPController::B_T_O_r(Vector3d p_0) { return Vector3d(r_B_O_r_ + B_R_O_r_ * p_0); }

Vector3d DualPandaCoSTPController::O_T_B_l(Vector3d p_B) { return Vector3d(O_R_B_l_ * (p_B - r_B_O_l_)); }

Vector3d DualPandaCoSTPController::O_T_B_r(Vector3d p_B) { return Vector3d(B_R_O_r_ * (p_B - r_B_O_r_)); }

Quaterniond DualPandaCoSTPController::B_T_O_l(Quaterniond q_0) {
  return Quaterniond(B_R_O_l_ * q_0.toRotationMatrix());
}

Quaterniond DualPandaCoSTPController::B_T_O_r(Quaterniond q_0) {
  return Quaterniond(B_R_O_r_ * q_0.toRotationMatrix());
}

Quaterniond DualPandaCoSTPController::O_T_B_l(Quaterniond q_B) {
  return Quaterniond(O_R_B_l_ * q_B.toRotationMatrix());
}

Quaterniond DualPandaCoSTPController::O_T_B_r(Quaterniond q_B) {
  return Quaterniond(O_R_B_r_ * q_B.toRotationMatrix());
}

std::pair<Vector3d, Quaterniond> DualPandaCoSTPController::GetLinkTransform(const std::string& link_name) {
  return mjpc::QueryBodyPoseEigen(mj_model_, mj_data_, link_name, mj_task_->GetBaseBodyName());
}

void DualPandaCoSTPController::init() {
  // Right arm --
  auto right_arm_pose = GetLinkTransform("panda0_link0");
  r_B_O_r_ = right_arm_pose.first;
  const Quaterniond& B_R_O_r = right_arm_pose.second;
  B_R_O_r_ = B_R_O_r.toRotationMatrix();
  O_R_B_r_ = B_R_O_r_.transpose();

  // Left arm --
  auto left_arm_pose = GetLinkTransform("panda1_link0");
  r_B_O_l_ = left_arm_pose.first;
  const Quaterniond& B_R_O_l = left_arm_pose.second;
  setWholeBodyDistanceThresh(0.2);
  CoSTPController::init(r_B_O_r_, B_R_O_r, r_B_O_l_, B_R_O_l);
  B_R_O_l_ = B_R_O_l.toRotationMatrix();
  O_R_B_l_ = B_R_O_l_.transpose();

  // Init dual controllers
  left_controller_.init("panda0_joint1", "panda0_end_effector");
  right_controller_.init("panda1_joint1", "panda1_end_effector");

  // Start
  start();
}

void DualPandaCoSTPController::start() {
  DUAL_PANDA_LOCK_CONTROL_MUTEX;
  state_ = ControlState::IDLE;
  Vector7d q_l = Eigen::Map<Vector7d>(left_controller_.getQ().data(), 7);
  Vector7d q_r = Eigen::Map<Vector7d>(right_controller_.getQ().data(), 7);
  q_d_ << q_l, q_r;
  reset(q_d_);
}

void DualPandaCoSTPController::initSwitchingControl() {
  qD_d_.setZero();
  reset(q_d_);
}

Vector14d DualPandaCoSTPController::update() {
  static const double t = 0.001;
  DUAL_PANDA_LOCK_CONTROL_MUTEX;
  switch (state_) {
    case ControlState::INIT_SWITCHING_CONTROL: {
      initSwitchingControl();
      state_ = ControlState::SWITCHING_CONTROL;
    }
    case ControlState::SWITCHING_CONTROL: {
      if (new_goal_) {
        fillBuffer(current_goal_);
        new_goal_ = false;
      }
      Vector14d q, qD, q_d_old, tau_ext;
      auto q_l = left_controller_.getQ();
      auto q_r = right_controller_.getQ();
      auto qD_l = left_controller_.getQD();
      auto qD_r = right_controller_.getQD();
      auto tau_ext_l = left_controller_.getTauExtHat();
      auto tau_ext_r = right_controller_.getTauExtHat();
      q << q_l, q_r;
      qD << qD_l, qD_r;
      tau_ext << tau_ext_l, tau_ext_r;
      q_d_old = q_d_;
      q_d_ = followTrajectory(tau_ext, velocity_ / 0.9);
      qD_d_ = (q_d_ - q_d_old) * 1000;
      break;
    }
    case ControlState::JOINT_MOTION:
    case ControlState::IDLE: {
      Vector14d q;
      auto q_l = left_controller_.getQ();
      auto q_r = right_controller_.getQ();
      q << q_l, q_r;
      calculateControlPreliminaries(q);
      break;
    }
  }

  Vector14d tau_d = Vector14d::Zero();
  //tau_d.head<7>() = left_controller_.control(q_d_.head<7>());
  //tau_d.tail<7>() = right_controller_.control(q_d_.tail<7>());

  // Next point ready -> start bimanual planning
  auto* bimanual_planner = dynamic_cast<PandaBimanualPlanner*>(planner());
  if (readyForNextGoal() && !new_goal_handled_) {
    bimanual_planner->PlanCallback(abs_pos());
    new_goal_handled_ = true;
  }
  return q_d_;
}

void DualPandaCoSTPController::targetPoseCallback(const Vector3d& position) {
  DUAL_PANDA_LOCK_CONTROL_MUTEX;
  switch (state_) {
    case ControlState::IDLE: {
      state_ = ControlState::INIT_SWITCHING_CONTROL;
      break;
    }
    case ControlState::INIT_SWITCHING_CONTROL:
    case ControlState::SWITCHING_CONTROL: {
      break;
    }
    default: {
      print(1, "target pose discarded as robot is in wrong state.");
      return;
    }
  }
  current_goal_ = position;
  if (!readyForNextGoal()) {
    print("Discarding point: ", current_goal_.transpose(),
          " as buffer is busy. Please wait until receiving a position"
          " response before sending the next trajectory point.");
  }
  new_goal_ = true;
  new_goal_handled_ = false;
}

bool DualPandaCoSTPController::jointMotionCallback(double v, const Vector14d& goal) {
  if (v > 2) {
    print("Maximum speed of motion is 2 rad/s. Aborting");
    return false;
  }
  Vector14d q;
  {
    DUAL_PANDA_LOCK_CONTROL_MUTEX;
    Vector7d q_l(left_controller_.getQ()), q_r(right_controller_.getQ());
    q << q_l, q_r;
    state_ = ControlState::JOINT_MOTION;
    reset(q);
  }
  {
    DUAL_PANDA_LOCK_CONTROL_MUTEX;
    JointMotionGenerator::generateC1Trajectory(q_d_, qD_d_, v, q, goal);
    state_ = ControlState::IDLE;
  }
  return true;
}
} // namespace mjpc
