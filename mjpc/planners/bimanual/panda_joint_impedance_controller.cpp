// Copyright (c) 2020 Lehrstuhl für Robotik und Sysstemintelligenz, TU München
#include "mjpc/planners/bimanual/panda_joint_impedance_controller.h"

#include <cmath>

// Abseil
#include "absl/container/btree_set.h"
#include "absl/types/span.h"

// mjpc
#include "mjpc/utilities.h"
#include "mjpc/planners/bimanual/damping_design.h"
#include "mjpc/planners/bimanual/bimanual_planner.h"

namespace mjpc {
template <int N>
void PandaJointImpedanceController<N>::init(const std::string& first_joint_name,
                                            const std::string& endtip_name) {
  first_joint_name_ = first_joint_name;
  endtip_name_ = endtip_name;
  first_dof_id_ = mj_model_->dof_jntid[QueryJointId(mj_model_, first_joint_name.c_str())];
}

template
void PandaJointImpedanceController<7>::init(const std::string& first_joint_name,
                                            const std::string& endtip_name);

template <int N>
typename PandaJointImpedanceController<N>::MatrixNd PandaJointImpedanceController<N>::jointDampingDesign(
    const MatrixNd& stiffness,
    const MatrixNd& damping_ratio,
    const MatrixNd& inertia) {
  // Compute Cartesian Mass inverse, previously adding motor intertia B
  VectorNd b, k_T; // NOLINT (readability-identifier-naming)
  // Motor mass matrix
  MatrixNd inertia_hat;
  b << 0.6057, 0.6057, 0.4625, 0.4625, 0.2055, 0.2055, 0.2055;
  k_T << 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2.5;
  MatrixNd B, K_T, I_K_T_inv; // NOLINT (readability-identifier-naming)
  B = b.asDiagonal();
  K_T = k_T.asDiagonal();
  I_K_T_inv = MatrixNd::Identity() + K_T;
  I_K_T_inv = I_K_T_inv.inverse();
  inertia_hat << inertia + I_K_T_inv * B;
  return panda_controllers_impl::factorizationDesign<N>(stiffness, damping_ratio, inertia_hat);
}

template <int N>
typename PandaJointImpedanceController<N>::VectorNd PandaJointImpedanceController<N>::control(
    const VectorNd& q_d, const VectorNd& qD_d) {
  std::array<double, N * N> inertia_array{};
  MjuFullMatrix(mj_model_, inertia_array.data(), &mj_data_->qM[mj_model_->dof_Madr[first_dof_id_]],
                  first_dof_id_, N);
  const MatrixNd inertia = Eigen::Map<const MatrixNd>(inertia_array.data(), N, N);
  const VectorNd coriolis = Eigen::Map<const VectorNd>(&mj_data_->qfrc_bias[first_dof_id_], N);
  const VectorNd q = getQ();
  const VectorNd qD = getQD();
  static MatrixNd damping_ratio = MatrixNd::Identity() * 0.8;
  VectorNd stiffness;
  mju_copy(stiffness.data(), &mj_model_->jnt_stiffness[mj_model_->dof_jntid[first_dof_id_]], N);
  if (stiffness.isZero()) {
    stiffness = VectorNd{600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0};
  }

  MatrixNd designed_damping =
      jointDampingDesign(stiffness_scale_ * stiffness.asDiagonal(), damping_ratio, inertia);
  const VectorNd tau_d =
      coriolis + stiffness_scale_ * stiffness.asDiagonal() * (q_d - q) + designed_damping * (qD_d - qD);
  return tau_d;
}

template
typename PandaJointImpedanceController<7>::VectorNd PandaJointImpedanceController<7>::control(
    const VectorNd& q_d, const VectorNd& qD_d);

template <int N>
void PandaJointImpedanceController<N>::setStiffnessScale(const double& stiffness_scale) {
  stiffness_scale_ = stiffness_scale;
}

template <int N>
double PandaJointImpedanceController<N>::limitQD(VectorNd qD_d, const double& v_cart) {
  double scale = 1;
  // [jac] for [endtip_name_]
  std::vector<double> jac(6 * N);
  mj_jacBody(mj_model_, mj_data_, &jac[0], &jac[3 * N],
             mj_task_->QueryBodyId(endtip_name_.c_str()));

  // jac -> J
  // Refer to dm_robotics utils.h/.cc
  const auto J = Eigen::Map<const Eigen::Matrix<double, 6, N>>(jac.data(), 6, N);

  VectorNd q = getQ();
  double v = (J * qD_d).template head<3>().norm();
  if (v_cart > 0) {
    if (v > v_cart) {
      scale = v_cart / v;
      qD_d *= scale;
    }
  }
  VectorNd qD_max;
  qD_max << 2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61;
  qD_max *= 0.99;
  double max_rate = 1;
  for (int i = 0; i < N; ++i) {
    if (std::abs(qD_d[i]) > qD_max[i]) {
      max_rate = std::max(max_rate, std::abs(qD_d[i]) / qD_max[i]);
    }
  }
  return scale / max_rate;
}
} // namespace mjpc
