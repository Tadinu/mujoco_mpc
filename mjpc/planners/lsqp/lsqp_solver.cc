#include "mjpc/planners/lsqp/lsqp_solver.h"

// mjpc
#include "mjpc/core/mjpc_common.h"
#include "mjpc/tasks/lsqp/lsqp.h"


namespace mjpc {
void LsqpSolver::SetFrameTaskTarget(const mjData* data, LsqpFrameTask* task,
                                    const char* target_mocap_name) const {
  const auto* target_mocap_quat = mjpc::QueryBodyMocapQuat(model_, data, target_mocap_name);
  const auto* target_mocap_pos = mjpc::QueryBodyMocapPos(model_, data, target_mocap_name);
  const auto T_wt = LsqpSE3(target_mocap_quat, target_mocap_pos);
  task->SetTarget(T_wt);
}
} // end namespace mjpc