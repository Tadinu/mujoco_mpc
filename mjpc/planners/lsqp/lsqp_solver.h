#pragma once

#include <mujoco/mujoco.h>

// mjpc
#include "mjpc/planners/lsqp/lsqp_base_task.h"
#include "mjpc/planners/lsqp/lsqp_config.h"
#include "mjpc/planners/lsqp/lsqp_posture_task.h"
#include "mjpc/planners/lsqp/lsqp_relative_frame_task.h"
#include "mjpc/planners/lsqp/lsqp_limit.h"
#include "mjpc/planners/cross_entropy/planner.h"

namespace mjpc {
class Lsqp;

class LsqpSolver {
public:
  LsqpSolver() = default;

  explicit LsqpSolver(const mjModel* model, Lsqp* lsqp, MjOwnerAppType owner_type) :
    model_(model),
    lsqp_task_(lsqp),
    owner_type_(owner_type) {
  }

  void Init(const mjData* data, int ndofs);
  std::vector<double> Solve(mjData* data);

  std::vector<double> DefaultControlInputs() const {
    return std::vector(config_.ndofs(), 0.);
  }

  LsqpConfig config() const {
    return config_;
  }

private:
  const mjModel* model_ = nullptr;
  Lsqp* lsqp_task_ = nullptr;
  MjOwnerAppType owner_type_ = MjOwnerAppType::MJAPP;
  LsqpSE3 T_ee_initial_;

  // Solver
  LsqpConfig config_;
  std::vector<LsqpBaseTask*> subtasks_;
  LsqpFrameTask end_effector_subtask_;
  LsqpPostureTask posture_subtask_;
  std::vector<LsqpRelativeFrameTask> finger_subtasks_;
  std::vector<LsqpLimitPtr> config_limits_;
  void SetFrameTaskTarget(mjData* data, LsqpFrameTask* task, const char* target_mocap_name) const;
};

using LsqpSolverPtr = std::shared_ptr<LsqpSolver>;
} // end namespace mjpc