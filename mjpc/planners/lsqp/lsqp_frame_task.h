#pragma once

// mjpc
#include "mjpc/planners/lsqp/lsqp_base_task.h"
#include "mjpc/planners/lsqp/lsqp_util.h"
#include "mjpc/planners/lsqp/lsqp_so3.h"
#include "mjpc/planners/lsqp/lsqp_se3.h"

namespace mjpc {
class LsqpFrameTask : public LsqpBaseTask {
public:
  LsqpFrameTask() = default;

  LsqpFrameTask(std::string name, const mjModel* model, std::string frame_name, int frame_type,
                Eigen::VectorXd position_cost,
                Eigen::VectorXd orientation_cost,
                double gain = 1.0, double lm_damping = 1.0) :
    LsqpBaseTask(std::move(name), model, {}, gain, lm_damping),
    frame_name_(std::move(frame_name)),
    frame_type_(frame_type),
    position_cost_(std::move(position_cost)),
    orientation_cost_(std::move(orientation_cost)) {
    is_frame_task_ = true;
    k_ = 6;
    if (position_cost_.size() == 1) {
      position_cost_ = Eigen::VectorXd::Constant(3, position_cost_[0]);
    }

    if (orientation_cost_.size() == 1) {
      orientation_cost_ = Eigen::VectorXd::Constant(3, orientation_cost_[0]);
    }
    cost_.resize(k_);
    cost_ << position_cost_, orientation_cost_;
  }

  ~LsqpFrameTask() override = default;

  virtual SE3 GetFrameTransform(mjData* data, const LsqpConfig& config) const {
    return config.GetTransformFrameToWorld(data, frame_name_, frame_type_);
  }

  void SetTarget(const SE3& target) {
    target_transform_ = target;
  }

  virtual void SetTargetFromConfig(mjData* data, const LsqpConfig& config) {
    SetTarget(GetFrameTransform(data, config));
  }

  Eigen::VectorXd ComputeError(mjData* data, const LsqpConfig& config) const override {
    return target_transform_.Minus(GetFrameTransform(data, config));
  }

  Eigen::MatrixXd ComputeJac(mjData* data, const LsqpConfig& config) const override {
    const Eigen::MatrixXd frame_jac = config.GetFrameJacobian(data, frame_name_, frame_type_);
    const SE3 current_frame_transf = GetFrameTransform(data, config);

    const SE3 new_target_transf = target_transform_.Inverse() * current_frame_transf;
    return -new_target_transf.JacLog() * frame_jac;
  }

protected:
  std::string frame_name_;
  int frame_type_ = -1;
  Eigen::VectorXd position_cost_;
  Eigen::VectorXd orientation_cost_;

  // Transform: target->base
  SE3 target_transform_;
};
}