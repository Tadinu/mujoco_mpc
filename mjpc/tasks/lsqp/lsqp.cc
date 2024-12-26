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

void Lsqp::TransitionLocked(mjModel* model, mjData* data) {
  //! Critical, so [lsqp_planner_] can query precise latest Mj data]
  if (lsqp_planner_) {
    lsqp_planner_->RefreshMj(model, data);
  }

  // Init mocaps
  if (!mocaps_inited_) {
    InitMocaps();
    mocaps_inited_ = true;
  }
}

void Lsqp::ResidualFn::Residual(const mjModel* model, const mjData* data, double* residual) const {
  int counter = 0;

  for (auto i = 0; i < 4; ++i) {
    residual[counter++] = 0;
  }

  // Sanity check
  CheckSensorDim(model, counter);
}
} // namespace mjpc
