#pragma once

#include <stdexcept>
#include <iostream>
#include <utility>
#include <vector>

// Eigen
#include <Eigen/Dense>

// MuJoCo
#include <mujoco/mujoco.h>

// qp_solver_collection
#include <qp_solver_collection/QpSolverCollection.h>

// mjpc
#include "mjpc/planners/lsqp/lsqp_se3.h"
#include "mjpc/utilities.h"

namespace mjpc {
class LsqpConfig {
public:
  LsqpConfig() = default;

  LsqpConfig(const mjModel* model, int ndofs,
             const QpSolverCollection::QpSolverType qp_solver_type =
                 QpSolverCollection::QpSolverType::QuadProg)
    : model_(model), ndofs_(ndofs),
      qp_solver_(QpSolverCollection::allocateQpSolver(qp_solver_type)) {
    if (ndofs != model->nu) {
      throw MjpcError::customized("Currently only supporting [ndofs == model->nu]",
                                  "ndofs: " + std::to_string(ndofs) + "# model->nu: "
                                  + std::to_string(model->nu));
    }
  }

  const mjModel* MjModel() const { return model_; }
  int nv() const { return model_->nv; }
  int nq() const { return model_->nq; }
  int ndofs() const { return ndofs_; }

  std::shared_ptr<QpSolverCollection::QpSolver> QpSolver() const {
    return qp_solver_;
  }

  bool CheckJointValues(const double* q, size_t jnts_num, double tol = 1e-2) {
    assert(model_->njnt >= jnts_num);
    for (int jnt_id = 0; jnt_id < jnts_num; ++jnt_id) {
      if (model_->jnt_type[jnt_id] == mjJNT_FREE || !model_->jnt_limited[jnt_id]) {
        continue;
      }

      double qval = q[jnt_id];
      double qmin = model_->jnt_range[jnt_id * 2];
      double qmax = model_->jnt_range[jnt_id * 2 + 1];
      if (qval < qmin - tol || qval > qmax + tol) {
        return false;
      }
    }
    return true;
  }

  bool CheckJointLimits(const mjData* data, double tol = 1e-2, bool safety_break = true) const {
    for (int jnt = 0; jnt < model_->njnt; ++jnt) {
      if (model_->jnt_type[jnt] == mjJNT_FREE || !model_->jnt_limited[jnt]) continue;

      const int qposAdr = model_->jnt_qposadr[jnt];
      const double qval = data->qpos[qposAdr];
      const int qvelAdr = model_->jnt_dofadr[jnt];
      const double qvel = data->qvel[qvelAdr];
      const double qmin = model_->jnt_range[jnt * 2];
      const double qmax = model_->jnt_range[jnt * 2 + 1];

      static constexpr auto MAX_QVEL = M_PI;
      const bool invalid_config = (qval < qmin - tol || qval > qmax + tol) || (
                                    std::abs(qvel) > MAX_QVEL);
      if (invalid_config) {
        std::cerr << "Warning: Joint " << jnt << " out of limits " << qval << " vs [" << qmin - tol << ","
            << qmax + tol << "]" << "- qvel: " << qvel << std::endl;
        if (safety_break) {
          return false;
        }
      }
    }
    return true;
  }

  Eigen::MatrixXd GetFrameJacobian(const mjData* data, const std::string& frameName, int frameType) const {
    int frameId = mj_name2id(model_, frameType, frameName.c_str());
    if (frameId < 0) {
      throw std::invalid_argument("Invalid frame: " + frameName);
    }

    // NOTE: Since [mj_jacBody, mj_jacSite] all use nv() implicitly, ndofs() cannot be used here!
    const int nv = this->nv();
    std::vector<mjtNum> jacBuffer(6 * nv, 0);
    auto* jacPtr = jacBuffer.data();

    if (frameType == mjOBJ_BODY) {
      mj_jacBody(model_, data, jacPtr, jacPtr + 3 * nv, frameId);
    } else if (frameType == mjOBJ_SITE) {
      mj_jacSite(model_, data, jacPtr, jacPtr + 3 * nv, frameId);
    } else {
      throw std::invalid_argument("Unsupported frame type");
    }

    // jac(i, j) = jacPtr[i * nv + j] where i: [0->5], j: [0->nv-1]
    Eigen::Map<Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::RowMajor>> jac(jacPtr, 6, nv);

    // MuJoCo jacobians have a frame of reference centered at the local frame but
    // aligned with the world frame. To obtain a jacobian expressed in the local
    /// frame, aka body jacobian, we need to left-multiply by A[T_fw].
    auto* frame_quat = (frameType == mjOBJ_BODY)
                         ? mjpc::QueryBodyQuat(data, frameId)
                         : mjpc::QuerySiteQuat(data, frameId);
    return SE3(SO3(frame_quat).Inverse()).Adjoint() * jac;
  }

  SE3 GetTransformFrameToWorld(const mjData* data, const std::string& frameName, int frameType) const {
    const int frameId = mj_name2id(model_, frameType, frameName.c_str());
    if (frameId < 0) {
      throw std::invalid_argument("Invalid frame: " + frameName);
    }

    const auto pos = (frameType == mjOBJ_BODY)
                       ? mjpc::QueryBodyPosEigen(data, frameId, false)
                       : (frameType == mjOBJ_SITE)
                       ? mjpc::QuerySitePosEigen(data, frameId)
                       : Eigen::Vector3d::Zero();
    mjtNum* quat = (frameType == mjOBJ_BODY)
                     ? mjpc::QueryBodyQuat(data, frameId, false)
                     : (frameType == mjOBJ_SITE)
                     ? mjpc::QuerySiteQuat(data, frameId)
                     : const_cast<mjtNum*>(mjpc::ROTATION_IDENTITY);
    return {SO3(quat), pos};
  }

  SE3 GetTransform(const mjData* data, const std::string& frameName, int frameType,
                   const std::string& refName, int refType) const {
    const auto transform_frame_to_world = GetTransformFrameToWorld(data, frameName, frameType);
    //transform_frame_to_world.PrintSelf(frameName);
    const auto transform_ref_to_world = GetTransformFrameToWorld(data, refName, refType);
    //transform_ref_to_world.PrintSelf(refName);
    return transform_ref_to_world.Inverse() * transform_frame_to_world;
  }

private:
  const mjModel* model_ = nullptr;
  // NOTE: mjData* should be provided by caller, eg as running in a thread of rollouts

  // Active dofs, typically ones of robots only, excluding dynamic objs that also have dofs
  int ndofs_ = 0;
  std::shared_ptr<QpSolverCollection::QpSolver> qp_solver_ = nullptr;
};

struct LsqpConstraint {
  Eigen::MatrixXd G; // Inequality constraint matrix
  Eigen::VectorXd h; // Inequality constraint vector

  bool Inactive() const {
    return !(G.size() && h.size());
  }
};
} // end namespace mjpc