#pragma once

#include "mjpc/planners/lsqp/lsqp_base_task.h"
#include "mjpc/planners/lsqp/lsqp_util.h"

namespace mjpc {
class LsqpPostureTask : public LsqpBaseTask {
public:
  LsqpPostureTask() = default;

  LsqpPostureTask(std::string name, const mjModel* model, const Eigen::VectorXd& cost, double gain = 1.0,
                  double lm_damping = 1.0) :
    LsqpBaseTask(std::move(name), model, cost, gain, lm_damping),
    target_q_(Eigen::VectorXd::Zero(model->nq)),
    v_ids_(GetFreeJointDims(model).second) {
    k_ = model->nv;
    SetCost(cost);
  }

  ~LsqpPostureTask() override = default;

  void SetCost(const Eigen::VectorXd& cost) {
    if (cost.size() == 1) {
      cost_ = Eigen::VectorXd::Constant(k_, cost[0]);
    } else {
      if (cost_.size() != k_) {
        cost_ = Eigen::VectorXd::Zero(k_);
      }
      mju_copy(cost_.data(), cost.data(), k_);
    }
  }

  bool Empty() const override {
    return (target_q_.size() == 0);
  }

  void SetTarget(Eigen::VectorXd target) {
    target_q_ = std::move(target);
  }

  Eigen::VectorXd ComputeError(const LsqpConfig& config) const override {
    if (Empty()) {
      throw std::runtime_error("`target_q_` is empty");
    }

    // NOTE: mj_differentiatePos calculates qpos2 ⊖ qpos1.
    Eigen::VectorXd qvel = Eigen::VectorXd::Zero(config.nv());
    mj_differentiatePos(config.MjModel(), qvel.data(), 1.0, config.MjData()->qpos, target_q_.data());
    mjpc::ResetEigenVector(qvel, v_ids_);
    return qvel;
  }

  Eigen::MatrixXd ComputeJac(const LsqpConfig& config) const override {
    if (Empty()) {
      throw std::runtime_error("`target_q_` is empty");
    }

    // !NOTE: Must declare [jac]'s type explicitly here for [setZero()] to compile
    Eigen::MatrixXd jac = -Eigen::MatrixXd::Identity(config.nv(), config.nv());
    for (int v_id : v_ids_) {
      jac.col(v_id).setZero();
    }
    return jac;
  }

private:
  Eigen::VectorXd target_q_;
  std::vector<int> v_ids_;
};
}