// Copyright (c) 2020 Lehrstuhl für Robotik und Sysstemintelligenz, TU München
#pragma once

#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "mjpc/planners/bimanual/controller_type.h"
#include "mjpc/planners/bimanual/costp_controller.h"
#include "mjpc/planners/bimanual/panda_joint_impedance_controller.h"

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Matrix<double, 8, 1> Vector8d;
typedef Eigen::Matrix<double, 14, 1> Vector14d;
typedef Eigen::Matrix<double, 7, 7> Matrix7d;
using Eigen::Matrix3d;
using Eigen::Quaterniond;

namespace mjpc {
enum class ControlState { JOINT_MOTION, IDLE, SWITCHING_CONTROL, INIT_SWITCHING_CONTROL };

class DualPandaCoSTPController : public CoSTPController {
public:
  DualPandaCoSTPController(mjModel* model, mjData* data, Task* task) : CoSTPController(model, data, task),
                                                                       left_controller_(model, data, task),
                                                                       right_controller_(model, data, task) {
  };
  void init();
  void start();
  Vector14d update();

  // To be renamed to [goalCallback]
  void targetPoseCallback(const Vector3d& position);
  bool jointMotionCallback(double v, const Vector14d& goal);
  void initSwitchingControl();
  void computeContactWrench();
  std::pair<Vector3d, Quaterniond> GetLinkTransform(const std::string& link_name);
  Vector6d transformWrenchFromOToB(/*state, */ const Eigen::Matrix3d& B_R_O);
  Vector3d B_T_O_l(Vector3d p_0);
  Vector3d B_T_O_r(Vector3d p_0);
  Vector3d O_T_B_l(Vector3d p_B);
  Vector3d O_T_B_r(Vector3d p_B);
  Quaterniond B_T_O_l(Quaterniond q_0);
  Quaterniond B_T_O_r(Quaterniond q_0);
  Quaterniond O_T_B_l(Quaterniond q_B);
  Quaterniond O_T_B_r(Quaterniond q_B);

private:
  Matrix3d B_R_O_l_, B_R_O_r_, O_R_B_l_, O_R_B_r_;
  Vector3d r_B_O_l_, r_B_O_r_;
  std::recursive_mutex control_mutex_;
#define DUAL_PANDA_LOCK_CONTROL_MUTEX std::lock_guard<std::recursive_mutex> lock(control_mutex_);
  Vector14d dq_, q_d_, qD_d_;
  Vector3d current_goal_;
  std::vector<ghostplanner::cfplanner::Obstacle> obstacles_;
  double velocity_ = 0.2;
  PandaJointImpedanceController<7> left_controller_;
  PandaJointImpedanceController<7> right_controller_;
  ControlState state_ = ControlState::IDLE;
  bool new_goal_ = false, new_goal_handled_ = false;
};

using DualPandaCoSTPControllerPtr = std::shared_ptr<DualPandaCoSTPController>;
} // namespace mjpc
