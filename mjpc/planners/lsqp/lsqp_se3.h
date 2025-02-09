#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cassert>
#include <cmath>

// mjpc
#include "mjpc/planners/lsqp/lsqp_common.h"
#include "mjpc/planners/lsqp/lsqp_so3.h"

// Ref: https://arxiv.org/pdf/1812.01537
namespace mjpc {
// Static Constant for Identity Element
static const Eigen::Matrix<double, 7, 1> IDENTITY_WXYZ_XYZ = (
      Eigen::VectorXd(7) << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).
    finished();

class LsqpSE3 : public LsqpMatrixLieGroup<LsqpSE3> {
public:
  // Static constants defining group dimensions
  static constexpr int MATRIX_DIM = 4;
  static constexpr int PARAMS_DIM = 7;
  static constexpr int DIM = 3;
  static constexpr int TANGENT_DIM = 6;

  LsqpSE3() = default;
  // Construct using an SO(3) object for rotation and a 3D translation vector
  LsqpSE3(LsqpSO3 rotation_part, Eigen::Vector3d translation_part = Eigen::Vector3d::Zero())
    : rotation_(std::move(rotation_part)), translation_(std::move(translation_part)) {
  }

  LsqpSE3(const Eigen::Vector3d& translation) : LsqpSE3(LsqpSO3(), translation) {
  }

  LsqpSE3(const double data[7] /* rotation[4] + translation[3]*/)
    : rotation_(LsqpSO3(&data[0])), translation_(&data[4]) {
  }

  LsqpSE3(const double rotation[4], const double translation[3])
    : rotation_(LsqpSO3(rotation)), translation_(translation) {
  }

  // Construct using parameters [qw, qx, qy, qz, x, y, z]
  LsqpSE3(const Eigen::VectorXd& parameters)
    : rotation_(LsqpSO3(parameters.head<4>())),
      translation_(parameters.tail<3>()) {
    assert(parameters.size() == PARAMS_DIM);
  }

  void PrintSelf(const std::string& label = "") const override {
    mjpc::print(label, "pos:", translation_.transpose(), "quat(WXYZ): ", rotation_.Parameters().transpose());
  }

  // Group dimensions
  int MatrixDim() const override {
    return MATRIX_DIM;
  }

  int ParametersDim() const override {
    return PARAMS_DIM;
  }

  int Dim() const override {
    return DIM;
  }

  int TangentDim() const override {
    return TANGENT_DIM;
  }

  LsqpSO3 Rotation() const { return rotation_; }
  Eigen::Vector3d Translation() const { return translation_; }

  LsqpSE3 Identity() const override {
    return LsqpSE3(rotation_.Identity(), Eigen::Vector3d::Zero());
  }

  // Factory method: Create group element from transformation matrix
  LsqpSE3 FromMatrix(const Eigen::MatrixXd& matrix) const override {
    assert(matrix.rows() == MATRIX_DIM && matrix.cols() == MATRIX_DIM);
    Eigen::Matrix3d rotation_matrix = matrix.block<3, 3>(0, 0);
    Eigen::Vector3d translation_vector = matrix.block<3, 1>(0, 3);
    return LsqpSE3(rotation_.FromMatrix(rotation_matrix), translation_vector);
  }

  // Factory method: Uniformly sample group element
  LsqpSE3 SampleUniform() const override {
    return LsqpSE3(rotation_.SampleUniform(), Eigen::Vector3d::Random());
  }

  // Access parameters [qw, qx, qy, qz, x, y, z]
  Eigen::VectorXd Parameters() const override {
    Eigen::VectorXd params(PARAMS_DIM);
    params.head<4>() = rotation_.Parameters();
    params.tail<3>() = translation_;
    return params;
  }

  // Return 4x4 transformation matrix
  Eigen::MatrixXd AsMatrix() const override {
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3, 3>(0, 0) = rotation_.AsMatrix();
    transform.block<3, 1>(0, 3) = translation_;
    return transform;
  }

  // Apply group action (transform a 3D point)
  Eigen::VectorXd Apply(const Eigen::VectorXd& target) const override {
    assert(target.size() == DIM);
    return rotation_.Apply(target) + translation_; // R * p + t
  }

  // Multiply two SE(3) transformations
  LsqpSE3 Multiply(const LsqpSE3& other) const override {
    return LsqpSE3(rotation_ * other.rotation_, Apply(other.translation_));
  }

  // Inverse of SE(3)
  LsqpSE3 Inverse() const override {
    LsqpSO3 inv_rotation = rotation_.Inverse();
    Eigen::Vector3d inv_translation = -(inv_rotation.Apply(translation_));
    return LsqpSE3(std::move(inv_rotation), std::move(inv_translation));
  }

  // Normalize the SE(3) element (ensures valid rotation)
  LsqpSE3 Normalize() const override {
    return LsqpSE3(rotation_.Normalize(), translation_);
  }

  // Exponential map (tangent space -> SE(3))
  LsqpSE3 Exp(const Eigen::VectorXd& tangent) const override {
    assert(tangent.size() == TANGENT_DIM);
    static LsqpSO3 so3;

    // Extract rotation and translation parts from the tangent vector
    const Eigen::Vector3d upsilon = tangent.head<3>();
    const Eigen::Vector3d theta = tangent.tail<3>();
    const Eigen::Matrix3d skew_omega = Skew(theta);
    const double theta_sq = theta.squaredNorm();
    const bool use_taylor = theta_sq < 1e-10;
    const double safe_theta_sq = use_taylor ? 1.0 : theta_sq;
    const double safe_theta = std::sqrt(theta_sq);

    const LsqpSO3 R = so3.Exp(theta);
    Eigen::Matrix3d V;
    if (use_taylor) {
      V = R.AsMatrix();
    } else {
      V = Eigen::Matrix3d::Identity()
          + (1.0 - std::cos(safe_theta)) / theta_sq * skew_omega
          + (safe_theta - std::sin(safe_theta)) / (safe_theta_sq * safe_theta) * (skew_omega * skew_omega);
    }

    return LsqpSE3(R, V * upsilon);
  }

  // Logarithm map (SE(3) -> tangent space)
  Eigen::VectorXd Log() const override {
    const Eigen::Vector3d omega = rotation_.Log(); // SO(3) logarithm
    const Eigen::Matrix3d skew_omega = Skew(omega);

    const double theta_sq = omega.squaredNorm();
    const bool use_taylor = theta_sq < 1e-10;
    const double safe_theta_sq = use_taylor ? 1.0 : theta_sq;
    const double safe_theta = std::sqrt(safe_theta_sq);
    const double half_theta_safe = 0.5 * safe_theta;
    const Eigen::Matrix3d skew_omega_norm = skew_omega * skew_omega;

    Eigen::Matrix3d V_inv = Eigen::Matrix3d::Identity();
    if (use_taylor) {
      V_inv = Eigen::Matrix3d::Identity() - (0.5 * skew_omega) + (skew_omega_norm / 12.0);
    } else {
      V_inv = Eigen::Matrix3d::Identity()
              - 0.5 * skew_omega
              + ((1 - (safe_theta * std::cos(half_theta_safe) / (2.0 * std::sin(half_theta_safe)))) /
                 safe_theta_sq) * skew_omega_norm;
    }

    Eigen::VectorXd tangent(TANGENT_DIM);
    tangent.head<3>() = V_inv * translation_; // Translation part
    tangent.tail<3>() = omega; // Rotation part
    return tangent;
  }

  // Compute adjoint matrix
  Eigen::MatrixXd Adjoint() const override {
    Eigen::MatrixXd adj = Eigen::MatrixXd::Zero(TANGENT_DIM, TANGENT_DIM);
    const Eigen::Matrix3d R = rotation_.AsMatrix();
    adj.block<3, 3>(0, 0) = R; // Top-left
    adj.block<3, 3>(0, 3) = Skew(translation_) * R; // Top-right
    adj.block<3, 3>(3, 3) = R; // Bottom-right
    return adj;
  }

  // Jacobians for SE(3)
  Eigen::MatrixXd LeftJac(const Eigen::VectorXd& tangent) const override {
    assert(tangent.size() == TANGENT_DIM);

    // Extract translational (upsilon) and rotational (omega) components
    Eigen::Vector3d upsilon = tangent.head<3>();
    Eigen::Vector3d theta = tangent.tail<3>();

    // Compute the rotational left Jacobian for SO(3)
    static LsqpSO3 so3;
    Eigen::Matrix3d J_R = so3.LeftJac(theta);

    // Compute the Q matrix
    Eigen::Matrix3d Q = GetQ(tangent);

    // Construct the 6x6 left Jacobian matrix
    Eigen::MatrixXd J_ljac = Eigen::MatrixXd::Zero(TANGENT_DIM, TANGENT_DIM);
    J_ljac.block<3, 3>(0, 0) = J_R; // Top-left: rotational Jacobian
    J_ljac.block<3, 3>(3, 0) = Q; // Bottom-left: Q coupling matrix
    J_ljac.block<3, 3>(3, 3) = J_R; // Bottom-right: rotational Jacobian

    return J_ljac;
  }

  Eigen::MatrixXd LeftJacInverse(const Eigen::VectorXd& tangent) const override {
    assert(tangent.size() == TANGENT_DIM);

    // Extract translational (upsilon) and rotational (omega) components
    Eigen::Vector3d theta = tangent.tail<3>();
    if (theta.dot(theta) < 1e-10) {
      return Eigen::MatrixXd::Identity(TANGENT_DIM, TANGENT_DIM);
    }

    // Compute the rotational left Jacobian inverse for SO(3)
    static LsqpSO3 so3;
    Eigen::Matrix3d J_R_inv = so3.LeftJacInverse(theta);

    // Compute the Q matrix
    Eigen::Matrix3d Q = GetQ(tangent);

    // Construct the 6x6 left Jacobian inverse
    Eigen::MatrixXd J_ljacinv = Eigen::MatrixXd::Zero(TANGENT_DIM, TANGENT_DIM);
    J_ljacinv.block<3, 3>(0, 0) = J_R_inv; // Top-left: inverse rotational Jacobian
    J_ljacinv.block<3, 3>(3, 0) = -J_R_inv * Q * J_R_inv; // Bottom-left: coupling
    J_ljacinv.block<3, 3>(3, 3) = J_R_inv; // Bottom-right: inverse rotational Jacobian
    return J_ljacinv;
  }

private:
  LsqpSO3 rotation_;
  Eigen::Vector3d translation_ = Eigen::Vector3d::Zero();

  static Eigen::Matrix3d GetQ(const Eigen::VectorXd& tangent) {
    const Eigen::Vector3d theta = tangent.tail<3>();
    double theta_sq = theta.squaredNorm();
    constexpr double A = 0.5;
    double B, C, D;
    if (theta_sq < 1e-10) {
      B = (1.0 / 6.0) + (1.0 / 120.0) * theta_sq;
      C = -(1.0 / 24.0) + (1.0 / 720.0) * theta_sq;
      D = -(1.0 / 60.0);
    } else {
      const double theta_norm = theta.norm();
      const double sin_t = std::sin(theta_norm);
      const double cos_t = std::cos(theta_norm);
      B = (theta_norm - sin_t) / (theta_sq * theta_norm);
      C = (1.0 - theta_sq / 2.0 - cos_t) / (theta_sq * theta_sq);
      D = ((2.0 * theta_norm) - (3.0 * sin_t) + (theta_norm * cos_t)) / (
            2.0 * theta_sq * theta_sq * theta_norm);
    }

    const auto V = Skew(tangent.head<3>());
    const auto W = Skew(theta);
    const auto VW = V * W;
    const auto WV = VW.transpose();
    const auto WVW = WV * W;
    const auto VWW = VW * W;
    return (
      A * V
      + B * (WV + VW + WVW)
      - C * (VWW - VWW.transpose() - 3 * WVW)
      + D * (WVW * W + W * WVW)
    );
  }
};

using SE3 = LsqpSE3;
} // end namespace mjpc