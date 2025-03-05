#pragma once

// mjpc
#include "mjpc/planners/lsqp/lsqp_base_task.h"

namespace mjpc {
class LsqpCoMTask : public LsqpBaseTask {
public:
  LsqpCoMTask() = delete;

  LsqpCoMTask(std::string name, mjModel* model, std::string frame_name, int frame_type,
              const Eigen::VectorXd& position_cost,
              const Eigen::VectorXd& orientation_cost,
              double gain = 1.0, double lm_damping = 1.0) :
    LsqpBaseTask(std::move(name), model, Eigen::VectorXd::Zero(model->nv), gain, lm_damping) {
    k_ = 3;
    target_CoM_ = Eigen::VectorXd::Zero(k_);
  }

  ~LsqpCoMTask() override = default;

  bool Empty() const override {
    return target_CoM_.size() == 0;
  }

  void SetTarget(Eigen::VectorXd target_q) {
    target_CoM_ = std::move(target_q);
  }

  virtual void SetTargetFromConfig(mjData* data, const LsqpConfig& config) {
    SetTarget(mjpc::PosToEigen(&data->subtree_com[1], k_));
  }

  Eigen::VectorXd ComputeError(mjData* data, const LsqpConfig& config) const override {
    if (Empty()) {
      throw std::runtime_error("`target_CoM_` is empty");
    }

    std::vector<double> res(k_, 0);
    mju_sub(res.data(), &data->subtree_com[1], target_CoM_.data(), k_);
    return mjpc::PosToEigen(res.data(), k_);
  }

  Eigen::MatrixXd ComputeJac(mjData* data, const LsqpConfig& config) const override {
    if (Empty()) {
      throw std::runtime_error("`target_CoM_` is empty");
    }

    std::vector<double> jacBuffer(k_ * config.nv(), 0);
    mj_jacSubtreeCom(config.MjModel(), data, jacBuffer.data(), 1);
    return mjpc::ArrayToEigenMatrix(jacBuffer.data(), k_, config.nv(), true);
  }

protected:
  Eigen::VectorXd target_CoM_;
};
}