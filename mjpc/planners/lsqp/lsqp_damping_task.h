#pragma once

#include "lsqp_posture_task.h"
#include "mjpc/planners/lsqp/lsqp_posture_task.h"
#include "mjpc/planners/lsqp/lsqp_util.h"

namespace mjpc {
class LsqpDampingTask : public LsqpPostureTask {
public:
  LsqpDampingTask() = default;

  LsqpDampingTask(std::string name, mjModel* model, const Eigen::VectorXd& cost) :
    LsqpPostureTask(std::move(name), model, cost, /* gain */0.0, /*damping*/0.0) {
    target_q_ = Eigen::Map<Eigen::VectorXd>(model->qpos0, nq_);
  }

  ~LsqpDampingTask() override = default;

private:
  Eigen::VectorXd target_q_;
};
}