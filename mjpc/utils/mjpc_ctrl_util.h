#pragma once

#include <any>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <variant>
#include <memory>
#include <cstring>

// MuJoCo
#include <mujoco/mujoco.h>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// mjpc
#include "mjpc/utilities.h"
#include "mjpc/utils/mjpc_core_util.h"

namespace mjpc {
// CONTROL ------------------
//
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;

// Differential inverse kinematics function
inline Eigen::VectorXd DiffIk(const Eigen::MatrixXd& J, // Jacobian matrix
                              const Vector6d& vee_desired, // Desired end-effector velocity
                              double damping = 1e-4) {
  // Damping factor for regularization
  assert(J.rows() == 6);

  // Solve the system
  // https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
  // https://eigen.tuxfamily.org/dox/group__TopicSparseSystems.html
#if 1
  // Regularized pseudoinverse (Tikhonov regularization or ridge regression)
  // => Help stabilize the movement if in case of being around singularities (Jac gets ill-conditioned, almost losing rank).
  // -> Prevent the solution from getting too large.
  // (J*Jt + diag)⋅x=vee_desired
  const Matrix6d regularization = damping * Matrix6d::Identity();

  auto Jt = J;
  Jt.transposeInPlace();
  return Jt * (J * Jt + regularization).ldlt().solve(vee_desired);
#else
  return mjpc::robust_inv(J, damping) * vee_desired;
#endif
}

inline Eigen::VectorXd DiffNullspace(const Eigen::MatrixXd& J, // Jacobian matrix
                                     const Vector6d& vee_desired,
                                     const VectorXd& delta_q,
                                     double damping = 1e-4) {
  // Nullspace control biasing joint velocities towards the home configuration
  Eigen::VectorXd dq = DiffIk(J, vee_desired, damping);
  VectorXd Kn(delta_q.size());
  Kn << 10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0;
  for (auto i = 0; i < Kn.size(); ++i) {
    Kn[i] *= delta_q[i];
  }
  static const auto eye = MatrixXd::Identity(delta_q.size(), delta_q.size());
  dq += (eye - pinv(J) * J) * Kn;

  // Clamp maximum joint velocity.
  static constexpr double max_angvel = 0.785;
  const auto dq_abs_max = dq.cwiseAbs().maxCoeff();
  if (dq_abs_max > max_angvel) {
    dq *= max_angvel / dq_abs_max;
  }
  return dq;
}

inline Eigen::VectorXd ControlDiff(const mjModel* model, const mjData* data,
                                   const char* base_name, const char* ee_name, const char* target_name,
                                   const mjtNum* cur_qpos,
                                   mjtNum integration_dt,
                                   bool nullspace = false) {
  // NOTE: Here, we assume [ee_name] is both body & site name
  //const int base_id = query_body_id(model, base_name);
  const int ee_id = mjpc::QueryBodyId(model, ee_name);
  const int target_id = mjpc::QueryBodyId(model, target_name);
  auto* target_pos = mjpc::QueryBodyPos(data, target_id);
  auto* ee_pos = mjpc::QueryBodyPos(data, ee_id);
  auto* target_quat = mjpc::QueryBodyQuat(data, target_id);
  auto* ee_quat = mjpc::QueryBodyQuat(data, ee_id);

  const int nv = model->nv;
  // construct chain and sparse Jacobians
  std::vector<mjtNum> jac(6 * nv, 0);
#if 0
  std::vector<int>  chain(6 * nv, 0);

  // NOTE: May also try [mj_jacSparseSimple(model, data, jac, jac+3*nnz, ee_pos, ee_id, nv, base_id)]
  // get sparse body Jacobian structure
  const int nnz = mjpc::mjpc_bodyChain(model, chain.data(), ee_id, base_id);
  assert(NV == nnz);
  mjpc::mjpc_jacSparse(model, data, jac, jac + 3 * nnz, ee_pos, ee_id, nv, chain.data());
#else
  mj_jac(model, data, jac.data(), jac.data() + 3 * nv, ee_pos, ee_id);
#endif

  // jac -> J
  // Refer to dm_robotics utils.h/.cc
  const auto J = Eigen::Map<const Eigen::MatrixXd>(jac.data(), 6, nv);

  const double kpos = nullspace ? 0.95 : 1;
  const double krot = nullspace ? 0.95 : 1;

  Vector6d vee_desired = Vector6d::Zero();
  // Linear component
  mju_sub3(vee_desired.data(), target_pos, ee_pos);
  mju_scl3(vee_desired.data(), vee_desired.data(), kpos);
  // Angular component
  mju_subQuat(vee_desired.data() + 3, target_quat, ee_quat);
  mju_scl3(vee_desired.data() + 3, vee_desired.data() + 3, krot);
  vee_desired /= integration_dt;

  // Compute joint velocities
  assert(nv == model->nq);
  const auto key_qpos = mjpc::QueryKeyJointPositions(model, "home");
  const Eigen::VectorXd delta_q = Eigen::Map<const Eigen::VectorXd>(key_qpos.data(), nv) -
                                  Eigen::Map<const Eigen::VectorXd>(cur_qpos, nv);
  return nullspace ? DiffNullspace(J, vee_desired, delta_q) : DiffIk(J, vee_desired);
}
} // namespace mj_app
