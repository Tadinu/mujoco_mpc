#pragma once

// mjpc
#include "mjpc/planners/lsqp/lsqp_frame_task.h"
#include "mjpc/planners/lsqp/lsqp_util.h"
#include "mjpc/planners/lsqp/lsqp_so3.h"
#include "mjpc/planners/lsqp/lsqp_se3.h"

namespace mjpc {
class LsqpRelativeFrameTask : public LsqpFrameTask {
public:
  LsqpRelativeFrameTask() = default;

  LsqpRelativeFrameTask(std::string name, const mjModel* model, std::string frame_name, int frame_type,
                        std::string base_name, int base_type,
                        Eigen::VectorXd position_cost,
                        Eigen::VectorXd orientation_cost,
                        double gain = 1.0, double lm_damping = 1.0) :
    LsqpFrameTask(std::move(name), model, std::move(frame_name), frame_type,
                  std::move(position_cost), std::move(orientation_cost), gain, lm_damping),
    base_name_(std::move(base_name)),
    base_type_(base_type) {
  }

  ~LsqpRelativeFrameTask() override = default;

  SE3 GetFrameTransform(const mjData* data, const LsqpConfig& config) const override {
    return config.GetTransform(data, frame_name_, frame_type_, base_name_, base_type_);
  }

  void SetTargetFromConfig(mjData* data, const LsqpConfig& config) override {
    SetTarget(GetFrameTransform(data, config));
  }

  Eigen::VectorXd ComputeError(const mjData* data, const LsqpConfig& config) const override {
    return GetFrameTransform(data, config).Minus(target_transform_);
  }

  Eigen::MatrixXd ComputeJac(const mjData* data, const LsqpConfig& config) const override {
    const Eigen::MatrixXd frame_jac = config.GetFrameJacobian(data, frame_name_, frame_type_);
    //mjpc::print(frame_jac);
    const Eigen::MatrixXd base_jac = config.GetFrameJacobian(data, base_name_, base_type_);
    //mjpc::print(base_jac);
    const SE3 current_frame_transf = GetFrameTransform(data, config);
    const SE3 new_target_transf = target_transform_.Inverse() * current_frame_transf;
    return new_target_transf.JacLog() *
           (frame_jac - current_frame_transf.Inverse().Adjoint() * base_jac);
  }

private:
  std::string base_name_;
  int base_type_ = -1;
};
}