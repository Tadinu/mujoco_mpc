#pragma once

#include <mujoco/mujoco.h>

// mjpc
#include "mjpc/planners/lsqp/lsqp_damping_task.h"
#include "mjpc/planners/lsqp/lsqp_base_task.h"
#include "mjpc/planners/lsqp/lsqp_config.h"
#include "mjpc/planners/lsqp/lsqp_posture_task.h"
#include "mjpc/planners/lsqp/lsqp_relative_frame_task.h"
#include "mjpc/planners/lsqp/lsqp_limit.h"
#include "mjpc/planners/cross_entropy/planner.h"

namespace mjpc {
class Lsqp;

class LsqpSolver : public BaseSolver {
  friend class Lsqp;

public:
  LsqpSolver() = default;

  explicit LsqpSolver(const mjModel* model, Lsqp* lsqp, MjOwnerAppType owner_type) : BaseSolver(),
    model_(model),
    lsqp_task_(lsqp),
    owner_type_(owner_type) {
  }

  std::vector<double> DefaultControlInputs() const {
    return std::vector(config_.ndofs(), 0.);
  }

  LsqpConfig Config() const {
    return config_;
  }

  std::vector<LsqpLimitPtr> ConfigLimits() const {
    return config_limits_;
  }

public:
  const mjModel* model_ = nullptr;
  Lsqp* lsqp_task_ = nullptr;
  MjOwnerAppType owner_type_ = MjOwnerAppType::MJAPP;
  LsqpSE3 T_wrist_initial_;

  // Solver
  LsqpConfig config_;
  std::vector<LsqpBaseTask*> subtasks_;
  std::vector<LsqpRelativeFrameTask> end_effector_subtasks_;
  LsqpPostureTask posture_subtask_;
  std::vector<LsqpRelativeFrameTask> finger_subtasks_;
  LsqpDampingTask damping_subtask_;
  std::vector<LsqpLimitPtr> config_limits_;
  void SetFrameTaskTarget(const mjData* data, LsqpFrameTask* task, const char* target_mocap_name) const;
};

using LsqpSolverPtr = std::shared_ptr<LsqpSolver>;
} // end namespace mjpc