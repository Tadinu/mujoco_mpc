#pragma once

#include <any>
#include <iostream>
#include <map>
#include <string>
#include <variant>
#include <memory>

// MuJoCo
#include <mujoco/mujoco.h>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// mjpc
#include "mjpc/utilities.h"
#include "mjpc/tasks/lsqp/lsqp.h"
#include "mjpc/utils/mjpc_core_util.h"
#include "mjpc/utils/mjpc_math_util.h"

namespace mjpc {
// CONTROL ------------------
//
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
static constexpr uint8_t JAC_ROWS_NUM = 6; // linearXYZ + rotXYZ

// Differential inverse kinematics function
// Ref: [kevinzakka]-https://github.com/kevinzakka/mjctrl/blob/main/diffik.py
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

// Ref: [kevinzakka]-https://github.com/kevinzakka/mjctrl/blob/main/diffik_nullspace.py
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
  dq += (eye - RobustInv(J) * J) * Kn;

  // Clamp maximum joint velocity.
  static constexpr double max_angvel = 0.785;
  const auto dq_abs_max = dq.cwiseAbs().maxCoeff();
  if (dq_abs_max > max_angvel) {
    dq *= max_angvel / dq_abs_max;
  }
  return dq;
}

// NOTE: To avoid dynamic memory allocation (safety) & clarity, do not return Eigen::MatrixXd here!
inline std::vector<mjtNum> MjCalculateJacobian(const mjModel* model, const mjData* data,
                                               const int ee_site_id,
                                               int dofs_num,
                                               const std::string& base_body_name = {}) {
  const auto& nv = model->nv;
  const auto jacr_adr = (JAC_ROWS_NUM / 2) * nv;
  assert(dofs_num <= nv);

  // Calculate Jacobian
  std::vector<mjtNum> jac(JAC_ROWS_NUM * nv, 0);
  if (base_body_name.empty()) {
    mj_jacSite(model, data, jac.data(), jac.data() + jacr_adr, ee_site_id);
  } else {
    // Construct chain: [base_body_name] -> [ee_body_name] and sparse Jacobian
    std::vector<int> chain(nv, 0);
    const int ee_body_id = mjpc::QueryBodyIdFromSite(model, ee_site_id);
    const int base_body_id = mjpc::QueryBodyId(model, base_body_name.data());
    assert(base_body_id >= 0);
#if 0
    std::vector<mjtNum> jac1(JAC_ROWS_NUM * nv, 0);
    std::vector<mjtNum> jac2(JAC_ROWS_NUM * nv, 0);
    mjpc::MjuJacDifPair(model, data, chain.data(), base_body_id, ee_body_id,
                        mjpc::QueryBodyPos(data, base_body_id),
                        mjpc::QuerySitePos(data, ee_site_id),
                        jac1.data(), jac2.data(), jac.data(),
                        jac1.data() + jacr_adr, jac2.data() + jacr_adr, jac.data() + jacr_adr);
#else
    dofs_num = mjpc::MjuBodyChain(model, chain.data(), ee_body_id, base_body_id);

    // Get sparse body Jacobian structure
    if (dofs_num > 0) {
      jac.resize(JAC_ROWS_NUM * dofs_num, 0);
      mjpc::MjuJacSparse(model, data, jac.data(), jac.data() + (JAC_ROWS_NUM / 2) * dofs_num,
                         mjpc::QuerySitePos(data, ee_site_id), ee_body_id, dofs_num,
                         chain.data(), base_body_id);
    } else {
      const std::string ee_body_name = mj_id2name(model, mjOBJ_BODY, ee_body_id);
      throw MjpcError::customized("[ControlDiff()] MjuBodyChain() failed",
                                  "base_body_name: " + base_body_name + ",id: " + std::to_string(base_body_id)
                                  + " & ee_body_name: " + ee_body_name + ",id: " + std::to_string(
                                      ee_body_id));
    }
#endif
  }

  // [jac] -> [out_jac]
  // NOTE: [out_jac] as static, TEMP HACK TO SOLVE UNKNOWN EIGEN MEMORY ISSUE, THIS IS NOT MULTITHREAD-FRIENDLY
  // https://eigen.tuxfamily.org/dox/TopicPitfalls.html
  const int jac_cols = base_body_name.empty() ? nv : dofs_num;
  if (jac_cols == dofs_num) {
    return jac;
  } else {
    // [jac_cols > dofs_num]
    std::vector<mjtNum> out_jac(JAC_ROWS_NUM * dofs_num, 0);
    for (auto i = 0; i < JAC_ROWS_NUM; ++i) {
      memcpy(out_jac.data() + i * dofs_num, &jac[i * jac_cols], dofs_num * sizeof(mjtNum));
    }
    return out_jac;
  }
}

// Ref: [kevinzakka]-https://github.com/kevinzakka/mjctrl/blob/main/diffik.py
inline Eigen::VectorXd ControlDiff(const mjModel* model, const mjData* data,
                                   const std::string& ee_site_name,
                                   const std::string& target_name,
                                   const mjtNum* cur_qpos,
                                   mjtNum integration_dt,
                                   const mjtNum* key_qpos = nullptr /* Only for nullspace control*/,
                                   const std::string& base_body_name = {}) {
  const int ee_site_id = mjpc::QuerySiteId(model, ee_site_name.c_str());
  const auto* ee_site_pos = mjpc::QuerySitePos(data, ee_site_id);
  const auto* ee_site_quat = mjpc::QuerySiteQuat(data, ee_site_id);
  const int target_id = mjpc::QueryBodyId(model, target_name.c_str());
  const auto* target_pos = mjpc::QueryBodyPos(data, target_id);
  const auto* target_quat = mjpc::QueryBodyQuat(data, target_id);
  const bool nullspace = (key_qpos != nullptr);

  // Jacobian: [base_body_name] -> [ee_site]
  const std::vector<mjtNum> jac = MjCalculateJacobian(model, data, ee_site_id, model->nv, base_body_name);
  const Eigen::MatrixXd J = mjpc::ArrayToEigenMatrix<JAC_ROWS_NUM>(jac.data(), jac.size() / JAC_ROWS_NUM);
  const int dofs_num = J.cols();

  const double kpos = nullspace ? 0.95 : 1;
  const double krot = nullspace ? 0.95 : 1;

  // Desired EE vel
  Vector6d vee_desired;
  // - Linear component
  auto* vee_lin_desired = vee_desired.data();
  mju_sub3(vee_lin_desired, target_pos, ee_site_pos);
  mju_scl3(vee_lin_desired, vee_lin_desired, kpos);

  // - Angular component
  auto* vee_ang_desired = vee_desired.data() + 3;
#if 1
  // NOTE: error_quat * ee_site_quat = target_quat (left multiplication)
  mjtNum error_quat[4];
  mju_negQuat(error_quat, ee_site_quat);
  mju_mulQuat(error_quat, target_quat, error_quat);
  mju_quat2Vel(vee_ang_desired, error_quat, integration_dt);
#else
  // NOTE: [mju_subQuat()] assumes right multiplication: ee_site_quat * error_quat = target_quat
  mju_subQuat(vee_ang_desired, target_quat, ee_site_quat);
  vee_desired /= integration_dt;
#endif
  mju_scl3(vee_ang_desired, vee_ang_desired, krot);

  // Compute joint velocities
  if (key_qpos) {
    const Eigen::VectorXd delta_q = Eigen::Map<const Eigen::VectorXd>(key_qpos, dofs_num) -
                                    Eigen::Map<const Eigen::VectorXd>(cur_qpos, dofs_num);
    return DiffNullspace(J, vee_desired, delta_q);
  } else {
    return DiffIk(J, vee_desired);
  }
}

// Ref: [kevinzakka]-https://github.com/kevinzakka/mjctrl/blob/main/opspace.py
inline Eigen::VectorXd ControlOSC(const mjModel* model, mjData* data,
                                  const std::string& ee_site_name, const std::string& target_name,
                                  const mjtNum* q0, // key qpos
                                  const mjtNum* cur_qpos,
                                  const mjtNum* cur_qvel,
                                  const mjtNum* cur_qfrc_bias,
                                  const VectorXd& Kp_null, // Impedance control gains
                                  mjtNum integration_dt,
                                  const std::string& base_body_name = {},
                                  bool gravity_compensation = false) {
  const auto nv = model->nv;
  const int ee_site_id = mjpc::QuerySiteId(model, ee_site_name.c_str());
  const int target_id = mjpc::QueryBodyId(model, target_name.c_str());
  const auto* target_pos = mjpc::QueryBodyPos(data, target_id);
  const auto* ee_site_pos = mjpc::QuerySitePos(data, ee_site_id);
  const auto* target_quat = mjpc::QueryBodyQuat(data, target_id);
  const auto* ee_site_quat = mjpc::QuerySiteQuat(data, ee_site_id);

  // Gains for the twist computation. These should be between 0 and 1. 0 means no
  // movement, 1 means move the end-effector to the target in one integration step.
  static constexpr double kpos = 0.95;
  // Gain for the orientation component of the twist computation. This should be
  // between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
  // orientation in one integration step.
  static constexpr double kori = 0.95;

  // Cartesian impedance control gains
  static const Vector3d impedance_pos = {100.0, 100.0, 100.0}; // [N/m]
  static const Vector3d impedance_ori = {50.0, 50.0, 50.0}; // [Nm/rad]
  static const Vector6d Kp = (Vector6d() << impedance_pos, impedance_ori).finished();

  // Damping ratio for both Cartesian and joint impedance control.
  static constexpr double damping_ratio = 1.0;
  // Compute damping and stiffness matrices
  static const Vector3d damping_pos = damping_ratio * 2 * impedance_pos.array().sqrt();
  static const Vector3d damping_ori = damping_ratio * 2 * impedance_ori.array().sqrt();

  static const Vector6d Kd = (Vector6d() << damping_pos, damping_ori).finished();
  const VectorXd Kd_null = damping_ratio * 2 * Kp_null.array().sqrt();

  // Desired EE twist
  Vector6d twist;
  // - Linear component
  auto* twist_lin = twist.data();
  mju_sub3(twist_lin, target_pos, ee_site_pos);
  mju_scl3(twist_lin, twist_lin, kpos);

  // - Angular component
  auto* twist_ang = twist.data() + 3;
#if 1
  // NOTE: error_quat * ee_site_quat = target_quat (left multiplication)
  mjtNum error_quat[4];
  mju_negQuat(error_quat, ee_site_quat);
  mju_mulQuat(error_quat, target_quat, error_quat);
  mju_quat2Vel(twist_ang, error_quat, integration_dt);
#else
  // NOTE: This assumes right multiplication: ee_site_quat * error_quat = target_quat
  mju_subQuat(twist_ang, target_quat, ee_site_quat);
  twist /= integration_dt;
#endif
  mju_scl3(twist_ang, twist_ang, kori);

  // Jacobian: [base_body_name] -> [ee_site]
  const auto dofs_num = Kp_null.size();
  const std::vector<mjtNum> jac =
      MjCalculateJacobian(model, data, ee_site_id, dofs_num, base_body_name);
  const Eigen::MatrixXd J = mjpc::ArrayToEigenMatrix<JAC_ROWS_NUM>(jac.data(), jac.size() / JAC_ROWS_NUM);
  const Eigen::MatrixXd Jt = J.transpose();
  const Eigen::VectorXd q0_eigen = mjpc::ArrayToEigen(q0, dofs_num);
  const Eigen::VectorXd cur_qpos_eigen = mjpc::ArrayToEigen(cur_qpos, dofs_num);
  const Eigen::VectorXd cur_qvel_eigen = mjpc::ArrayToEigen(cur_qvel, dofs_num);
  const Eigen::VectorXd cur_qfrc_bias_eigen = mjpc::ArrayToEigen(cur_qfrc_bias, dofs_num);

  // Compute the task-space inertia matrix (Mx)
  // NOTE: Must always use full M_full_inv(nv, nv) as the output in [mj_solveM]
  MatrixRowMajorXd M_full_inv(nv, nv);
  MatrixRowMajorXd M_inv;
  if (base_body_name.empty()) {
    mj_solveM(model, data, M_full_inv.data(), mjpc::MjuIdentityMatrix(nv).data(), dofs_num);
    M_inv = M_full_inv.block(0, 0, dofs_num, dofs_num);
  } else {
    mj_solveM(model, data, M_full_inv.data(), mjpc::MjuIdentityMatrix(nv).data(), nv);
    const int base_body_id = mjpc::QueryBodyId(model, base_body_name.data());
    const int base_dof_id = mjpc::QueryDofIdFromBody(model, base_body_id);
    M_inv = M_full_inv.block(base_dof_id, base_dof_id, dofs_num, dofs_num);
  }
  // [Mx]: The effective operational/task-space inertia matrix
  // -> Lambda as used in https://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
  const Eigen::MatrixXd Mx_inv = J * M_inv * Jt;
  Eigen::MatrixXd Mx;
  if (std::abs(Mx_inv.determinant()) >= 1e-2) {
    Mx = Mx_inv.inverse();
  } else {
    Mx = mjpc::Pinv(Mx_inv, 1e-2);
  }

  // Compute generalized forces
  Eigen::VectorXd tau = Jt * Mx * (Kp.cwiseProduct(twist) - Kd.cwiseProduct(J * cur_qvel_eigen));

  // Add joint task in nullspace
  const Eigen::MatrixXd Jbar = M_inv * Jt * Mx;
  const Eigen::VectorXd ddq = Kp_null.cwiseProduct(q0_eigen - cur_qpos_eigen) -
                              Kd_null.cwiseProduct(cur_qvel_eigen);
  tau += (Eigen::MatrixXd::Identity(dofs_num, dofs_num) - Jt * Jbar.transpose()) * ddq;

  // Add gravity compensation
  if (gravity_compensation) {
    tau += cur_qfrc_bias_eigen;
  }

  return tau;
}
} // namespace mj_app
