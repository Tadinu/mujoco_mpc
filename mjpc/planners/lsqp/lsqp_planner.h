#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <thread>

#include <mujoco/mujoco.h>

// mjpc
#include "mjpc/planners/lsqp/lsqp_config.h"
#include "mjpc/planners/planner.h"
#include "mjpc/utilities.h"
#include "mjpc/core/mjpc_common.h"
#include "mjpc/planners/lsqp/lsqp_posture_task.h"
#include "mjpc/planners/lsqp/lsqp_relative_frame_task.h"
#include "mjpc/planners/lsqp/lsqp_limit.h"
#include "mjpc/tasks/lsqp/lsqp.h"

#define MJPC_PLANNER_LSQP_DIFFIK_ENABLED (0)

namespace mjpc {
class Lsqp;

class LsqpPlanner : public Planner {
public:
  LsqpPlanner() = default;
  ~LsqpPlanner() override = default;

  explicit LsqpPlanner(MjOwnerAppType type) : Planner(type) {
  }

  std::vector<std::string> FingertipNames() const;
  std::map<std::string, std::vector<float>> FingertipRGBAs() const;
  std::string GetAttachmentPrefix() const;

  void InitLsqpEnv(const mjModel* model, mjData* data);
  void LsqpControl(bool position_ctrl = false);
  void RefreshMj(mjModel* model, mjData* data);

  // =========================================================================================================
  // MJPC-PLANNER IMPL --
  //
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  Lsqp* lsqp_task_ = nullptr;

  // initialize data and settings
  void Initialize(mjModel* model, const Task& task) override;

  void Allocate() override;

  // reset memory to zeros
  void Reset(int horizon, const double* initial_repeated_action = nullptr) override {
  }

  void SetState(const State& state) override {
  }

  const Trajectory* BestTrajectory() override { return trajectory_.get(); }

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override;

  void ClearTrace() override {
    const MjpcSharedMutexLock lock(policy_mutex_);
    trajectory_->trace.clear();
  }

  // planner-specific GUI elements
  void GUI(mjUI& ui) override {
  }

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift, int timer_shift, int planning,
             int* shift) override {
  }

  // return number of parameters optimized by planner
  int NumParameters() override { return 0; }

  // optimize nominal policy
  void OptimizePolicy(int horizon, ThreadPool& pool) override {
    const MjpcSharedMutexLock lock(policy_mutex_);
    LsqpControl();
  }

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override {
  }

  // set action from policy
  void ActionFromPolicy(double* action, const double* state, double time, bool use_previous) override;

protected:
  // mjpc
  std::shared_ptr<Trajectory> trajectory_ = nullptr;
  int dim_state_ = 0; // state
  int dim_state_derivative_ = 0; // state derivative
  int dim_action_ = 0; // action
  int dim_sensor_ = 0; // output (i.e., all sensors)
  int dim_max_ = 0; // maximum dimension
  mutable std::shared_mutex policy_mutex_;
  // [action_] is shared among policy motion planning threads.
  // NOTE: Using type as vector of primitive, CaSX is unclear why not well synch-protected yet.
  std::vector<double> action_;

  // lsqp
  LsqpConfig config_;
  std::vector<LsqpBaseTask*> subtasks_;
  LsqpFrameTask end_effector_task_;
  LsqpPostureTask posture_task_;
  std::vector<LsqpRelativeFrameTask> finger_tasks_;
  std::vector<LsqpLimitPtr> config_limits_;
  LsqpSE3 T_ee_prev_;

  void SetFrameTaskTarget(LsqpFrameTask* task, const char* target_mocap_name) const;
};

using LsqpPlannerPtr = std::shared_ptr<LsqpPlanner>;
} // end namespace mjpc