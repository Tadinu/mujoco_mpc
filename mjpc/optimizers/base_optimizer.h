#pragma once

// MuJoCo
#include <mujoco/mujoco.h>

// MJPC
#include "mjpc/planners/planner.h"
#include "mjpc/task.h"

class BaseOptimizer {
public:
  BaseOptimizer() = default;
  BaseOptimizer(mjModel* mj_model, mjData* mj_data, mjpc::Task* mj_task, mjpc::Planner* mj_planner)
      : mj_model_(mj_model), mj_data_(mj_data), mj_task_(mj_task), mj_planner_(mj_planner) {}
  virtual void optimize() = 0;

  virtual std::vector<double> opt_vals() const { return {}; }

protected:
  mjModel* mj_model_ = nullptr;
  mjData* mj_data_ = nullptr;
  mjpc::Task* mj_task_ = nullptr;
  mjpc::Planner* mj_planner_ = nullptr;
};
using BaseOptimizerPtr = std::shared_ptr<BaseOptimizer>;