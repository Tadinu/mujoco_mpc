// Copyright (c) 2020 Lehrstuhl für Robotik und Sysstemintelligenz, TU München
#pragma once

#include "mjpc/planners/bimanual/controller_type.h"

#include <atomic>
#include <Eigen/Dense>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

// DQ
#include <dqrobotics/DQ.h>
#include <dqrobotics/robot_modeling/DQ_CooperativeDualTaskSpace.h>
#include <dqrobotics/robot_modeling/DQ_Kinematics.h>
#include <dqrobotics/robot_modeling/DQ_SerialManipulator.h>

// mjpc
#include "mjpc/planners/bimanual/cf_agent.h"
#include "mjpc/planners/bimanual/obstacle.h"
#include "mjpc/planners/bimanual/controller_type.h"
#include "mjpc/planners/bimanual/obstacle.h"
#include "mjpc/planners/bimanual/trajectory_buffer.h"
#include "mjpc/core/mjpc_common.h"
#include "mjpc/utils/mjpc_core_util.h"
#include "mjpc/task.h"

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Vector8d = Eigen::Matrix<double, 8, 1>;
using Vector14d = Eigen::Matrix<double, 14, 1>;
using Matrix1d = Eigen::Matrix<double, 1, 1>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
using Matrix14d = Eigen::Matrix<double, 14, 14>;
using Matrix8d = Eigen::Matrix<double, 8, 8>;
using Matrix11d = Eigen::Matrix<double, 11, 11>;
using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector3d;

using DQ_robotics::DQ;
using DQ_robotics::DQ_CooperativeDualTaskSpace;
using DQ_robotics::DQ_Kinematics;
using DQ_robotics::DQ_SerialManipulator;
using ghostplanner::cfplanner::Obstacle;

// CoSTP: Cooperative Set-based Task-Priority control
namespace mjpc {
class CoSTPController {
public:
  CoSTPController() = default;
  CoSTPController(mjModel* model, mjData* data, Task* task);
  ~CoSTPController() = default;

  Planner* planner() const { return mj_task_ ? mj_task_->planner_ : nullptr; };

  void fillBuffer(const Vector3d& goal);
  Vector14d followTrajectory(const Vector14d& tau_ext, double vel);
  void init(const Vector3d& r_B_O_r, const Quaterniond& B_R_O_r, const Vector3d& r_B_O_l,
            const Quaterniond& B_R_O_l);
  void reset(const Vector14d& q);
  void setTasks(const std::unordered_map<ControllerType::Type, double>& gains);
  void setWholeBodyDistanceThresh(double distance_0);
  void setSwitching(bool switching);
  void calculateControlPreliminaries(const Vector14d& q);
  Vector8d dqrd() const { return dqrd_.q; };
  Vector8d dqad() const { return dqad_.q; };
  Vector8d rel_error() const { return rel_error_; };
  Vector3d abs_error() const { return abs_error_; };
  Vector3d rel_pos() const { return dqrd_.translation().q.segment(1, 3); };
  Vector3d abs_pos() const { return dqad_.translation().q.segment(1, 3); };
  double e_n() const { return e_n_; };
  double dot_s() const { return dot_s_; };
  double angle() const { return angle_; };
  bool readyForNextGoal() const { return ready_for_next_goal_; };

protected:
  mjModel* mj_model_ = nullptr;
  mjData* mj_data_ = nullptr;
  Task* mj_task_ = nullptr;
  // control variables and intermediate results
  Eigen::Matrix<double, 8, 14> AbsPJ_;
  Eigen::Matrix<double, 8, 14> RelPJ_;
  Eigen::Matrix<double, 6, 14> geomJac_;
  Vector14d q_c_ = Vector14d::Zero(), q_min_ = Vector14d::Zero(),
            q_max_ = Vector14d::Zero(),
            q_ = Vector14d::Zero(), qD_ = Vector14d::Zero();
  Vector8d rel_error_ = Vector8d::Zero();
  Vector3d abs_error_ = Vector3d::Zero(), rot_err_int_ = Vector3d::Zero(),
           goal_direction_ = Vector3d::Zero(),
           /*current goal*/ cg_ = Vector3d::Zero(),
           /*current goal buffer*/ cg_buf_ = Vector3d::Zero(),
           /*last goal*/ lg_ = Vector3d::Zero(),
           /*current nominal goal*/ current_ng_ = Vector3d::Zero(),
           /*last nominal goal*/last_ng_ = Vector3d::Zero(),
           /*current instantaneous goal*/current_ig_ = Vector3d::Zero(),
           /*next instantaneous goal*/ next_ig_ = Vector3d::Zero();
  double next_ng_ = 0, v_act_ = 0, v_goal_ = 0, v_max_buf_ = 0;
  double e_n_ = 0, dot_s_ = 0, angle_ = 0, catchup_reserve_ = 0.1;
  DQ dqrd_, dqad_, l_, lz_, ee_orientation_1_, r_, relp_;
  bool switching_ = true;
  std::atomic<bool> ready_for_next_goal_ = false;
  // body agents for collision avoidance (joints that will avoid collisions)
  // TODO this should be initialized from yaml file
  std::vector<std::pair<int, Eigen::Vector3d>> avoidance_poi_{{4, {0, 0, 0}}};
  ghostplanner::cfplanner::CfAgent body_avoidance_agent_ = {};
  // DQ dual Panda representation
  std::unique_ptr<DQ_SerialManipulator> franka_left_;
  std::unique_ptr<DQ_SerialManipulator> franka_right_;
  std::unique_ptr<DQ_CooperativeDualTaskSpace> dq_dual_panda_;
  // Controller configuration
  std::vector<ControllerType::Type> controller_types_ = {ControllerType::RELATIVE_POSE,
                                                         ControllerType::EE_TILT,
                                                         ControllerType::JOINT_LIMIT_AVOIDANCE,
                                                         ControllerType::ABSOLUTE_POSITION,
                                                         ControllerType::WHOLE_BODY_AVOIDANCE};
  std::unordered_map<ControllerType::Type, double> gains_ = {{ControllerType::RELATIVE_POSE, 0.005},
                                                             {ControllerType::EE_TILT, 0.1},
                                                             {ControllerType::JOINT_LIMIT_AVOIDANCE, 10.0},
                                                             {ControllerType::ABSOLUTE_POSITION, 1.0},
                                                             {ControllerType::WHOLE_BODY_AVOIDANCE, 0.0}
  };
  std::unique_ptr<TrajectoryBuffer> traj_buffer_ = std::make_unique<TrajectoryBuffer>(1);
  std::recursive_mutex traj_buffer_mutex_;
  //MoveitWholeBodyAvoidance mwba_;
  // control functions and tasks
  Vector14d control(const Vector14d& q, const Vector14d& qD,
                    const Vector3d& current_goal, const Vector14d& tau_ext, double vel);
  void positionControlImpl(MatrixXd& Jac, Vector14d& err, const Vector3d& cg,
                           const Vector3d& x_ff, const ControllerType::Type& ct);
  Vector3d calInstantaneousGoal();
  void calCurrentNominalGoal();
  void relativePoseControl(MatrixXd& Jac, Vector14d& err);
  void absolutePositionControl(MatrixXd& Jac, Vector14d& err,
                               const Vector3d& cg, double v_max);
  void EETiltControl(MatrixXd& Jac, Vector14d& err);
  void rotationalAdmittanceControl(MatrixXd& Jac, Vector14d& err,
                                   const Vector14d& tau_ext);
  void jointLimitAvoidanceControl(MatrixXd& Jac, Vector14d& err,
                                  const Vector14d& q);
  void wholeBodyCollisionAvoidance(MatrixXd& Jac, Vector14d& err,
                                   const Vector14d& q);
  void switchingOrder(const Vector14d& q, const Vector14d& qD);
  bool tiltFunc(double absangle, double tilt_min,
                double tilt_max) const;
  bool boolFunc(const Vector14d& q, const Vector14d& qD) const;
  bool boolJoint(const Vector14d& q, std::vector<int>& joints) const;
};

MatrixXd geomJ(const MatrixXd& absoluteposeJ, const DQ& absolutepose);
MatrixXd geomJ(const DQ_SerialManipulator& robot, const MatrixXd& poseJacobian,
               const VectorXd& q, int n);
} // namespace mjpc
