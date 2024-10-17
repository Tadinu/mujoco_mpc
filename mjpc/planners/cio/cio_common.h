#pragma once

#include <Eigen/Core>

struct CIOPose {
  CIOPose() = default;
  explicit CIOPose(const Eigen::Vector3d& position) { transf.block<3, 1>(0, 3) = position; }
  explicit CIOPose(const Eigen::Matrix3d& orientation) { transf.block<3, 3>(0, 0) = orientation; }
  Eigen::Matrix4d transf = Eigen::Matrix4d::Identity();
  Eigen::Vector3d position() const { return transf.block<3, 1>(0, 3); }
  Eigen::Matrix3d orientation() const { return transf.block<3, 3>(0, 0); }
  Eigen::Vector3d rpy() const { return orientation().eulerAngles(0, 1, 2); }
};

struct CIOVelocity {
  Eigen::Vector3d linear_vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d rot_vel = Eigen::Vector3d::Zero();
};

struct CIOAcceleration {
  Eigen::Vector3d linear_acc = Eigen::Vector3d::Zero();
  Eigen::Vector3d rot_acc = Eigen::Vector3d::Zero();
};

struct CIOContact {
  // Contact force
  Eigen::Vector3d f = Eigen::Vector3d::Zero();
  // Position of applied force in the frame of the manipulated object
  Eigen::Vector3d ro = Eigen::Vector3d::Zero();
  // [0,1]: Probability of being in contact
  double c = 0;
};

struct CIOContactState {};

struct CIOTrajectory {
  std::vector<CIOContactState> contact_states;
};

struct CIOObservation {
  int obj_id = -1;
  CIOPose pose;
  CIOVelocity vel;
  CIOContact contact;
};

struct CIOConfig {
  int K = 0;
  double delT = 0.001;
  double delT_phase = 0;
  double mass = 0;
  double mu = 0;
  std::vector<double> stage_weights;
  std::function<void()> init_traj;

  double steps_per_phase() const { return delT_phase / delT; }
  double T_steps() const { return K * steps_per_phase(); }
  double T_final() const { return K * delT_phase; }
};