#include "mjpc/tasks/lsqp/lsqp.h"

#include <string>

// mujoco
#include <mujoco/mujoco.h>

// mjpc
#include "mjpc/utilities.h"
#include "mjpc/planners/lsqp/lsqp_planner.h"

namespace mjpc {
void Lsqp::SetPlanner(Planner* planner) {
  Task::SetPlanner(planner);
  lsqp_planner_ = dynamic_cast<LsqpPlanner*>(planner);
}

void Lsqp::InitSolver(const MjOwnerAppType owner_type, int ndofs) {
  lsqp_solver_ = std::make_shared<LsqpSolver>(model_, this, owner_type);
  InitSolverConfigs(lsqp_solver_, data_, ndofs);
}
} // namespace mjpc
