#include "mjpc/planners/lsqp/lsqp_planner.h"
#include "mjpc/tasks/lsqp/lsqp.h"
#include "mjpc/planners/lsqp/lsqp_collision_limit.h"

namespace mjpc {
void LsqpPlanner::Initialize(mjModel* model, const Task& task) {
  model_ = model;
  action_dim_ = model->nu;
  prev_action_ = std::vector(action_dim_, 0.);

  // Init task fabrics
  if (task.IsFabricsSupported()) {
    InitTaskFabrics();
  }

  // Init task Lsqp
  if (task.IsLSQPSupported()) {
    lsqp_task_ = dynamic_cast<Lsqp*>(const_cast<Task*>(&task));
    InitTaskLsqp(model_, lsqp_task_->data_);
  }

  // Init [cem_delegate_]
  if (cem_delegate_) {
    cem_delegate_->Initialize(model, task);

    if (task.IsLSQPSupported()) {
      // Lsqp control callback -> embedded into [cem_delegate_]
      // Using bound lambda instead of std::bind for clarity & a bit faster
      cem_delegate_->SetControlCallback([this](double* policy_action, mjData* data) {
        return this->LsqpControl(policy_action, data);
      });

      // Overwrite [cem_delegate_]'s actions dim & limits
      cem_delegate_->SetActionDim(IIWA14_ALLEGRO_CEM_PARAMS_DIM);
      std::vector<double> limits;
      for (auto i = 0; i < IIWA14_ALLEGRO_CEM_PARAMS_DIM; ++i) {
        limits.push_back(-1);
        limits.push_back(1);
      }
      cem_delegate_->SetActionLimits(std::move(limits));
    }
  }
}

void LsqpPlanner::InitTaskLsqp(const mjModel* model, const mjData* data) {
  assert(lsqp_task_);

  // Init mocaps
  if (MjOwnerAppType::MJAPP == owner_type_) {
    lsqp_task_->InitMocaps();
  }
  // else: MJPC: Init is possible only in [TransitionLocked()] due to threading issue
}

void LsqpPlanner::Allocate() {
  // Allocate [cem_delegate_]'s trajectory
  if (cem_delegate_) {
    cem_delegate_->InitTrajectory();
    cem_delegate_->Allocate();
  }
}

// NOTE: This can run as a callback + possibly in a rollout thread, so this function must be kept agnostic,
// thread-safe. without dynamic allocation
std::vector<double> LsqpPlanner::LsqpControl(double* policy_action, mjData* data) {
  if (((MjOwnerAppType::MJPC == owner_type_) && !planning_on_) || (nullptr == lsqp_task_)) {
    return std::vector(action_dim_, 0.);
  }

  // Use [lsqp_task_]'s data if not provided
  const bool use_lsqp_task_data = (nullptr == data);
  if (use_lsqp_task_data) {
    data = lsqp_task_->data_;
  }

  // APPLY [policy_action], if outputted from the delegate planner (eg: [cem_delegate_])
  if (policy_action) {
    // NOTE: THIS MUST TAKE INTO ACCOUNT OF BOTH SCENARIOS WHEREBY TARGET-OBJ STAYS ON GROUND & ALREADY IN-HAND
    // Current target-obj pos
    double target_obj_pos[3];
    mju_copy3(target_obj_pos, mjpc::QueryBodyPos(model_, data, Lsqp::TARGET_OBJ_NAME));

    double target_goal_pos[3];
    mju_copy3(target_goal_pos, mjpc::QuerySitePos(model_, data, Lsqp::TARGET_OBJ_GOAL_NAME));

    double target_delta_pos[3];
    mju_sub3(target_delta_pos, target_goal_pos, target_obj_pos);
    mju_normalize3(target_delta_pos);

    // [EE-mocap pos] perturbation
    mjtNum new_ee_target_pos[3];
    mju_addScl3(new_ee_target_pos, target_obj_pos, target_delta_pos, std::abs(policy_action[0]));
    mjpc::SetBodyMocapPos(model_, data, Lsqp::EE_TARGET_NAME, new_ee_target_pos);
    if (use_lsqp_task_data) {
      const MjpcSharedMutexLock lock(policy_mutex_);
      mju_copy3(policy_ee_target_pos_, new_ee_target_pos);
    }

    // [EE-mocap quat]
    // Rot around Y 90
    mjtNum new_ee_target_quat[4];
    mju_axisAngle2Quat(new_ee_target_quat, (double[]){0, 1, 0}, M_PI_2);
#if 0
    mjtNum delta_ee_target_quat_Z[4];
    mju_axisAngle2Quat(delta_ee_target_quat_Z, (double[]){0, 0, 1},
                       M_PI * (1 + policy_action[1]));
    //mjtNum delta_ee_target_quat_X[4];
    //mju_axisAngle2Quat(delta_ee_target_quat_X, (double[]){1, 0, 0},
    //                  M_PI * (1 + policy_action[1]));
    mju_mulQuat(new_ee_target_quat, new_ee_target_quat, delta_ee_target_quat_Z);
    //mju_mulQuat(new_ee_target_quat, new_ee_target_quat, delta_ee_target_quat_X);
#endif
    mjpc::SetBodyMocapQuat(model_, data, Lsqp::EE_TARGET_NAME, new_ee_target_quat);

    // [Fingertip-mocaps] perturbation
#if MJPC_LSQP_FINGERS_OSC
    uint8_t i = 0;
    const auto hand_center_pos = lsqp_task_->GetHandCenterPos(data);
    for (const auto& fingertip : Lsqp::FINGERTIP_NAMES) {
      mjtNum new_finger_target_pos[3];
      mju_addScl3(new_finger_target_pos, hand_center_pos.data(),
                  lsqp_task_->initial_fingertips_direction[fingertip],
                  0.1 * policy_action[EE_CEM_PARAMS_DIM + (++i)]);
      mjpc::SetBodyMocapPos(model_, data, lsqp_task_->FingertipTargetBodyName(fingertip).c_str(),
                            new_finger_target_pos);
    }
#endif
  }

  // [Lsqp solver]: solve diff-ik
  // NOTE: This is made instance created per [LsqpControl()] to avoid dynamic allocation in threads,
  // which would cause sporadic crashes.
  auto lsqp_solver = LsqpSolver(model_, lsqp_task_, owner_type_);
  lsqp_solver.Init(data, action_dim_);
  auto ctrl = lsqp_solver.Solve(data);
#if !MJPC_LSQP_FINGERS_OSC
  for (uint8_t i = IIWA14_DOF; i < IIWA14_DOF + ALLEGRO_DOF; ++i) {
    const int jnt_id = model_->dof_jntid[i];
    const double low_lim = model_->jnt_range[2 * jnt_id];
    const double high_lim = model_->jnt_range[2 * jnt_id + 1];
    ctrl[i] = low_lim + (policy_action
                           ? std::abs(policy_action[EE_CEM_PARAMS_DIM + (i - IIWA14_DOF)])
                           : FabRandom::rand()) * (high_lim - low_lim);
  }
#endif
  if ((MjOwnerAppType::MJAPP == owner_type_) &&
      lsqp_solver.config().CheckJointValues(ctrl.data(), ctrl.size(), 0.1)) {
    mju_copy(data->ctrl, ctrl.data(), ctrl.size());
  }
  mjpc::print(ctrl);
  return ctrl;
}

void LsqpPlanner::Traces(mjvScene* scn) {
  if (cem_delegate_) {
    cem_delegate_->Traces(scn);
  }
  static constexpr float GREEN[] = {0.0, 1.0, 0.0, 1.0};
  AddConnector(scn ? scn : lsqp_task_->scene_, mjGEOM_ARROW, 0.005,
               policy_ee_target_pos_,
               mjpc::QueryBodyPos(model_, lsqp_task_->data_, Lsqp::TARGET_OBJ_NAME),
               GREEN);
#if 0
  std::vector<double> traces;
  {
    const MjpcSharedMutexLock lock(policy_mutex_);
    auto* traj = BestTrajectory();
    if (traj->trace.size() >= 6) {
      traces = traj->trace;
    }
  }

  static constexpr float GREEN[] = {0.0, 1.0, 0.0, 1.0};
  for (auto i = 0; (!traces.empty()) && (i < (traces.size() / 3) - 1); ++i) {
    AddConnector(scn ? scn : task_->scene_, mjGEOM_LINE, 5,
                       (mjtNum[]){traces[3 * i], traces[3 * i + 1], traces[3 * i + 2]},
                       (mjtNum[]){traces[3 * (i + 1)], traces[3 * (i + 1) + 1], traces[3 * (i + 1) + 2]},
                       GREEN);
  }
#endif
}

void LsqpPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  const MjpcSharedMutexLock lock(policy_mutex_);
  if (cem_delegate_) {
    cem_delegate_->OptimizePolicy(horizon, pool);
  }
}

void LsqpPlanner::ActionFromPolicy(double* action, const double* state, double time, bool use_previous) {
  const MjpcSharedMutexLock lock(policy_mutex_);
  if (cem_delegate_) {
    cem_delegate_->ActionFromPolicy(action, state, time, use_previous);
    // Convert [action] from [cem_delegate_]'s output space to control (data->ctrl) space
    auto act = LsqpControl(action);
    if (mju_isZero(act.data(), action_dim_)) {
      mju_copy(action, prev_action_.data(), action_dim_);
    } else {
      mju_copy(action, act.data(), action_dim_);
      mju_copy(prev_action_.data(), act.data(), action_dim_);
    }
  }
}
}
