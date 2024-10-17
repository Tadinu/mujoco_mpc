#pragma once

#include <chrono>
#include <random>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

struct CIOPose {
  CIOPose() = default;
  explicit CIOPose(const Eigen::Vector3d& position) : trans(position) {}
  explicit CIOPose(const Eigen::Quaterniond& orientation) : quat(orientation) {}
  Eigen::Vector3d position() const { return trans.vector(); }
  Eigen::Quaterniond orientation() const { return quat; }
  Eigen::Vector3d rpy() const {
    Eigen::Vector3d eulerAngles = quat.toRotationMatrix().eulerAngles(2, 1, 0);  // ZYX order
    return {eulerAngles(2), eulerAngles(1), eulerAngles(0)};
  }

  Eigen::Translation3d trans;
  Eigen::Quaterniond quat;
  void add_noise();
  constexpr int size() const { return (sizeof(trans) + sizeof(quat)) / sizeof(double); }
  std::vector<double> data() const {
    std::vector<double> out(size());
    std::memcpy(out.data(), trans.translation().data(), sizeof(trans));
    std::memcpy(out.data() + +(sizeof(trans) / sizeof(double)), quat.coeffs().data(), sizeof(quat));
    return out;
  }
};

struct CIOVelocity {
  Eigen::Vector3d linear_vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_vel = Eigen::Vector3d::Zero();
  void add_noise();
  constexpr int size() const { return (sizeof(linear_vel) + sizeof(angular_vel)) / sizeof(double); }
  std::vector<double> data() const {
    std::vector<double> out(size());
    std::memcpy(out.data(), linear_vel.data(), sizeof(linear_vel));
    std::memcpy(out.data() + (sizeof(linear_vel) / sizeof(double)), angular_vel.data(), sizeof(angular_vel));
    return out;
  }
};

struct CIOAcceleration {
  Eigen::Vector3d linear_acc = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_acc = Eigen::Vector3d::Zero();
  void add_noise();
  constexpr int size() const { return (sizeof(linear_acc) + sizeof(angular_acc)) / sizeof(double); }
  std::vector<double> data() const {
    std::vector<double> out(size());
    std::memcpy(out.data(), linear_acc.data(), sizeof(linear_acc));
    std::memcpy(out.data() + (sizeof(linear_acc) / sizeof(double)), angular_acc.data(), sizeof(angular_acc));
    return out;
  }
};

struct CIOGoal {
  CIOPose pose;
  CIOVelocity vel;
  // CIOAcceleration acc;
};

struct CIOContact {
  // Contact force
  Eigen::Vector3d f = Eigen::Vector3d::Zero();
  // Position of applied force in the frame of the manipulated object
  Eigen::Vector3d ro = Eigen::Vector3d::Zero();
  // [0,1]: Probability of being in contact
  double c = 0;

  Eigen::Vector3d pi_O_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d pi_H_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d e_O_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d e_H_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d e_dot_O_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d e_dot_H_ = Eigen::Vector3d::Zero();
  void add_noise();
  constexpr int size() const { return (sizeof(f) + sizeof(ro) + sizeof(c)) / sizeof(double); }
  std::vector<double> data() const {
    std::vector<double> out(size());
    std::memcpy(out.data(), f.data(), sizeof(f));
    std::memcpy(out.data() + (sizeof(f) / sizeof(double)), ro.data(), sizeof(ro));
    out.push_back(c);
    return out;
  }
};

struct CIOObservation {
  int obj_id = -1;
  CIOPose pose;
  CIOVelocity vel;
  CIOAcceleration acc;
  CIOContact contact;

  void add_noise();
  std::vector<double> data() const {
    std::vector<double> out(pose.size() + vel.size() + acc.size() + contact.size());
    std::memcpy(out.data(), pose.data().data(), sizeof(pose));
    std::memcpy(out.data() + (sizeof(pose) / sizeof(double)), vel.data().data(), sizeof(vel));
    std::memcpy(out.data() + (sizeof(pose) / sizeof(double)) + (sizeof(vel) / sizeof(double)),
                acc.data().data(), sizeof(acc));
    std::memcpy(out.data() + (sizeof(pose) / sizeof(double)) + (sizeof(vel) / sizeof(double)) +
                    (sizeof(acc) / sizeof(double)),
                contact.data().data(), sizeof(contact));
    return out;
  }
};

struct CIOStageWeight {
  double w_CI = 0;
  double w_physics = 0;
  double w_kinematics = 0;
  double w_task = 0;
};

struct CIOConfig {
  int K = 10;
  double delT = 0.001;
  double delT_phase = 0.5;
  double mass = 1.0;
  double mu = 0.9;      // Friction coefficient
  double lamb = 0.001;  // Regularization parameter

  std::vector<CIOStageWeight> stage_weights = {
      CIOStageWeight{.w_CI = 0.1, .w_physics = 0.1, .w_kinematics = 0.0, .w_task = 1.0},
      CIOStageWeight{.w_CI = 10.0, .w_physics = 1.0, .w_kinematics = 0.0, .w_task = 10.0}};
  std::function<void()> init_traj = nullptr;

  int steps_per_phase() const { return int(delT_phase / delT); }
  int T_steps() const { return K * steps_per_phase(); }
  double T_final() const { return K * delT_phase; }
};