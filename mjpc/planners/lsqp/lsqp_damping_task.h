#pragma once

#include "mjpc/planners/lsqp/lsqp_base_task.h"
#include "mjpc/planners/lsqp/lsqp_util.h"

namespace mjpc {
class LsqpDampingTask : public LsqpBaseTask {
public:
  LsqpDampingTask() = default;

  LsqpDampingTask(std::string name, mjModel* model, const Eigen::VectorXd& cost) :
    LsqpBaseTask(std::move(name), model, cost, /* gain */0.0, /*damping*/0.0) {
    target_q_ = Eigen::Map<Eigen::VectorXd>(model->qpos0, nq_);
  }

  ~LsqpDampingTask() override = default;

private:
  Eigen::VectorXd target_q_;
};
}