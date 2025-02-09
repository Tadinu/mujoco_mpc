#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cassert>
#include <cmath>

// MuJoCo
#include <mujoco/mujoco.h>

// mjpc
#include "mjpc/planners/lsqp/lsqp_common.h"
#include "mjpc/utils/mjpc_core_util.h"

// Ref: https://arxiv.org/pdf/1812.01537
namespace mjpc {
class LsqpSO3 : public LsqpMatrixLieGroup<LsqpSO3> {
public:
  // Static constants defining group dimensions
  static constexpr int MATRIX_DIM = 3;
  static constexpr int PARAMS_DIM = 4;
  static constexpr int DIM = 3;
  static constexpr int TANGENT_DIM = 3;
  LsqpSO3() = default;

  LsqpSO3(const Eigen::Quaterniond& quat) : quaternion_(quat) {
  }

  // NOTE: Don't use Quaterniond(Scalar* data) which assigns data directly to m_coeffs, which is stored in XYZW
  LsqpSO3(const Eigen::VectorXd& params) : quaternion_(mjpc::QuatToEigen(params.data())) {
    assert(params.size() == PARAMS_DIM);
  }

  LsqpSO3(const double quat[PARAMS_DIM]) : quaternion_(mjpc::QuatToEigen(quat)) {
  }

  void PrintSelf(const std::string& label = "") const override {
    mjpc::print(label, "quat(WXYZ): ", Parameters().transpose());
  }


  // Group dimensions
  int MatrixDim() const override { return MATRIX_DIM; }
  int ParametersDim() const override { return PARAMS_DIM; }
  int Dim() const override { return DIM; }
  int TangentDim() const override { return TANGENT_DIM; }

  Eigen::Quaterniond Quaternion() const {
    return quaternion_;
  }

  std::vector<double> Wxyz() const {
    return std::vector{quaternion_.w(),
                       quaternion_.x(),
                       quaternion_.y(),
                       quaternion_.z()
    };
  }

  // Factory method: Identity element
  LsqpSO3 Identity() const override {
    return {};
  }

  // Factory method: Create group element from matrix representation
  LsqpSO3 FromMatrix(const Eigen::MatrixXd& matrix) const override {
    assert(matrix.rows() == MATRIX_DIM && matrix.cols() == MATRIX_DIM);
    return LsqpSO3(Eigen::Quaterniond(matrix.topLeftCorner<3, 3>()));
  }

  // Factory method: Sample uniformly from the group
  LsqpSO3 SampleUniform() const override {
    return LsqpSO3(Eigen::Quaterniond::UnitRandom());
  }

  // Access the underlying quaternion parameters
  Eigen::VectorXd Parameters() const override {
    return Eigen::Vector4d{quaternion_.w(), quaternion_.x(), quaternion_.y(), quaternion_.z()};
  }

  // Return the 3x3 rotation matrix representation
  Eigen::MatrixXd AsMatrix() const override {
    return Quaternion().toRotationMatrix();
  }

  // Eq. 136 - Apply group action (rotate a 3D point/vector)
  Eigen::VectorXd Apply(const Eigen::VectorXd& target) const override {
    assert(target.size() == DIM);
    const auto padded_target = LsqpSO3(Eigen::Quaterniond(0, target[0], target[1], target[2]));
    const LsqpSO3 result = (*this) * padded_target * this->Inverse();
    return mjpc::PosToEigen(result.Wxyz().data() + 1, 3);
  }

  // Multiply two SO(3) elements (composition)
  LsqpSO3 Multiply(const LsqpSO3& other) const override {
#if 1
    mjtNum res[PARAMS_DIM];
    mju_mulQuat(res, Wxyz().data(), other.Wxyz().data());
    return LsqpSO3(res);
#else
    return LsqpSO3(quaternion_ * other.quaternion_);
#endif
  }

  // Inverse of the group element (rotation inverse)
  LsqpSO3 Inverse() const override {
    return LsqpSO3(quaternion_.inverse());
  }

  // Normalize the quaternion (to maintain valid rotation)
  LsqpSO3 Normalize() const override {
    return LsqpSO3(quaternion_.normalized());
  }

  // Eq 132 - Exponential map (tangent space -> SO(3))
  LsqpSO3 Exp(const Eigen::VectorXd& tangent) const override {
    assert(tangent.size() == TANGENT_DIM); // Ensure tangent is 3D

    const double theta_sq = tangent.squaredNorm();
    const double theta = std::sqrt(theta_sq);
    const double theta_pow_4 = std::pow(theta_sq, 2);
    const bool use_taylor = theta_sq < 1e-10;
    const double safe_theta = use_taylor ? 1.0 : theta;
    const double half_safe_theta = 0.5 * safe_theta;
    double re, img;
    if (use_taylor) {
      re = 1.0 - theta_sq / 8.0 + theta_pow_4 / 384.0;
      img = 0.5 - theta_sq / 48.0 + theta_pow_4 / 3840.0;
    } else {
      re = std::cos(half_safe_theta);
      img = std::sin(half_safe_theta) / safe_theta;
    }
    const auto xyz = img * tangent;
    return LsqpSO3((double []){re, xyz[0], xyz[1], xyz[2]});
  }

  // Eq 133 - Logarithm map (SO(3) -> tangent space) - Eq.133
  Eigen::VectorXd Log() const override {
    // Extract quaternion components
    const double qw = quaternion_.w();
    const Eigen::Vector3d qxyz(quaternion_.x(), quaternion_.y(), quaternion_.z());
    const double norm_sq = qxyz.squaredNorm();
    const bool use_taylor = norm_sq < 1e-10;
    const double norm_safe = use_taylor ? 1.0 : std::sqrt(norm_sq);
    const double w_safe = use_taylor ? qw : 1.0;
    const double atan_n_over_w = std::atan2((qw < 0) ? -norm_safe : norm_safe, std::abs(qw));

    double atan_factor = 0;
    if (use_taylor) {
      atan_factor = 2.0 / w_safe - 2.0 / 3.0 * norm_sq / std::pow(w_safe, 3);
    } else {
      if (std::abs(qw) < 1e-10) {
        const double scl = (qw > 0.0) ? 1.0 : -1.0;
        atan_factor = scl * M_PI / norm_safe;
      } else {
        atan_factor = 2.0 * atan_n_over_w / norm_safe;
      }
    }
    return atan_factor * qxyz;
  }

  // Eq 139 - Compute adjoint matrix (acts on tangent vectors; in SO(3) it's just the rotation matrix)
  Eigen::MatrixXd Adjoint() const override {
    return AsMatrix();
  }

  // Eq 145, 174 - Left Jacobian of SO(3)
  Eigen::MatrixXd LeftJac(const Eigen::VectorXd& tangent) const override {
    assert(tangent.size() == 3); // SO(3) tangent vectors are 3D

    const double theta_sq = tangent.squaredNorm();
    const double theta = std::sqrt(theta_sq);
    const Eigen::Matrix3d tangent_skew = Skew(tangent);
    double A, B;
    // Use Taylor series expansion for small theta
    if (theta < 1e-10) {
      const auto& t2 = theta_sq;
      A = (1.0 / 2.0) * (1.0 - t2 / 12.0 * (1.0 - t2 / 30.0 * (1.0 - t2 / 56.0)));
      B = (1.0 / 6.0) * (1.0 - t2 / 20.0 * (1.0 - t2 / 42.0 * (1.0 - t2 / 72.0)));
    } else {
      A = (1.0 - std::cos(theta)) / theta_sq;
      B = (theta - std::sin(theta)) / std::pow(theta, 3);
    }

    // Compute the Jacobian
    return Eigen::Matrix3d::Identity() + A * tangent_skew + B * (tangent_skew * tangent_skew);
  }

  // Inverse of Left Jacobian
  Eigen::MatrixXd LeftJacInverse(const Eigen::VectorXd& tangent) const override {
    assert(tangent.size() == TANGENT_DIM); // SO(3) tangent vectors are 3D

    // Theta: Tangent's norm
    const double theta_sq = tangent.squaredNorm();
    const double theta = std::sqrt(theta_sq);

    // Use Taylor series expansion for small values of theta
    double A;
    const auto& t2 = theta_sq;
    if (theta < 1e-10) {
      A = (1.0 / 12.0) * (1.0 + t2 / 60.0 * (1.0 + t2 / 42.0 * (1.0 + t2 / 40.0)));
    } else {
      // Full computation for general angles
      A = (1.0 / t2) * (1.0 - (theta * std::sin(theta) / (2.0 * (1.0 - std::cos(theta)))));
    }

    // Compute the inverse Jacobian
    const Eigen::Matrix3d tangent_skew = Skew(tangent);
    return Eigen::Matrix3d::Identity() - 0.5 * tangent_skew + A * (tangent_skew * tangent_skew);
  }

private:
  // Underlying quaternion representation [qw, qx, qy, qz]
  Eigen::Quaterniond quaternion_ = Eigen::Quaterniond::Identity();
};

using SO3 = LsqpSO3;
} // end namespace mjpc
