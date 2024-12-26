#pragma once

#include <stdexcept>
#include <iostream>
#include <utility>
#include <vector>

// Eigen
#include <Eigen/Dense>

// MuJoCo
#include <mujoco/mujoco.h>

// mjpc
#include "mjpc/planners/lsqp/lsqp_se3.h"
#include "mjpc/utilities.h"

namespace mjpc {
class LsqpConfig {
public:
  LsqpConfig() = default;

  LsqpConfig(const mjModel* model, mjData* data)
    : model_(model), data_(data) {
  }

  const mjModel* MjModel() const { return model_; }
  int nv() const { return model_->nv; }
  int nq() const { return model_->nq; }
  const mjData* MjData() const { return data_; }

  void RefreshMj(const mjModel* model, mjData* data) {
    model_ = model;
    data_ = data;
  }

  bool CheckJointValues(const Eigen::VectorXd& q, double tol = 1e-6) {
    assert(model_->njnt >= q.size());
    for (int jnt_id = 0; jnt_id < q.size(); ++jnt_id) {
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

  bool CheckJointLimits(double tol = 1e-6, bool safety_break = true) const {
    for (int jnt = 0; jnt < model_->njnt; ++jnt) {
      if (model_->jnt_type[jnt] == mjJNT_FREE || !model_->jnt_limited[jnt]) continue;

      const int qposAdr = model_->jnt_qposadr[jnt];
      const double qval = data_->qpos[qposAdr];
      const int qvelAdr = model_->jnt_dofadr[jnt];
      const double qvel = data_->qvel[qvelAdr];
      const double qmin = model_->jnt_range[jnt * 2];
      const double qmax = model_->jnt_range[jnt * 2 + 1];

      static constexpr auto MAX_QVEL = M_PI;
      const bool invalid_config = (qval < qmin - tol || qval > qmax + tol) || (std::abs(qvel) > MAX_QVEL);
      if (invalid_config) {
        std::cerr << "Warning: Joint " << jnt << " out of limits." << std::endl;
        if (safety_break) {
          return false;
        }
      }
      return true;
    }
  }

  Eigen::MatrixXd GetFrameJacobian(const std::string& frameName, int frameType) const {
    int frameId = mj_name2id(model_, frameType, frameName.c_str());
    if (frameId < 0) {
      throw std::invalid_argument("Invalid frame: " + frameName);
    }

    const int nv = model_->nv;
    Eigen::MatrixXd jac(6, nv);
    jac.setZero();
    std::vector<mjtNum> jacArr(6 * nv, 0);
    auto* jacPtr = jacArr.data();

    if (frameType == mjOBJ_BODY) {
      mj_jacBody(model_, data_, jacPtr, jacPtr + 3 * nv, frameId);
    } else if (frameType == mjOBJ_SITE) {
      mj_jacSite(model_, data_, jacPtr, jacPtr + 3 * nv, frameId);
    } else {
      throw std::invalid_argument("Unsupported frame type");
    }

    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < model_->nv; ++j) {
        jac(i, j) = jacPtr[i * model_->nv + j];
      }
    }

    // MuJoCo jacobians have a frame of reference centered at the local frame but
    // aligned with the world frame. To obtain a jacobian expressed in the local
    /// frame, aka body jacobian, we need to left-multiply by A[T_fw].
    auto* frame_quat = (frameType == mjOBJ_BODY)
                         ? mjpc::QueryBodyQuat(data_, frameId)
                         : mjpc::QuerySiteQuat(data_, frameId);
    return SE3(SO3(frame_quat).Inverse()).Adjoint() * jac;
  }

  SE3 GetTransformFrameToWorld(const std::string& frameName, int frameType) const {
    const int frameId = mj_name2id(model_, frameType, frameName.c_str());
    if (frameId < 0) {
      throw std::invalid_argument("Invalid frame: " + frameName);
    }

    const auto pos = (frameType == mjOBJ_BODY)
                       ? mjpc::QueryBodyPosEigen(data_, frameId, false)
                       : (frameType == mjOBJ_SITE)
                       ? mjpc::QuerySitePosEigen(data_, frameId)
                       : Eigen::Vector3d::Zero();
    mjtNum* quat = (frameType == mjOBJ_BODY)
                     ? mjpc::QueryBodyQuat(data_, frameId, false)
                     : (frameType == mjOBJ_SITE)
                     ? mjpc::QuerySiteQuat(data_, frameId)
                     : const_cast<mjtNum*>(mjpc::ROTATION_IDENTITY);
    return {SO3(quat), pos};
  }

  SE3 GetTransform(const std::string& frameName, int frameType,
                   const std::string& refName, int refType) const {
    const auto transform_frame_to_world = GetTransformFrameToWorld(frameName, frameType);
    //transform_frame_to_world.PrintSelf(frameName);
    const auto transform_ref_to_world = GetTransformFrameToWorld(refName, refType);
    //transform_ref_to_world.PrintSelf(refName);
    return transform_ref_to_world.Inverse() * transform_frame_to_world;
  }

private:
  const mjModel* model_ = nullptr;
  const mjData* data_ = nullptr;
};

struct LsqpConstraint {
  Eigen::MatrixXd G; // Inequality constraint matrix
  Eigen::VectorXd h; // Inequality constraint vector

  bool Inactive() const {
    return !(G.size() && h.size());
  }
};
} // end namespace mjpc