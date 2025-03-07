#include "mjpc/planners/lsqp/lsqp_planner.h"

#include "mjpc/utils/mjpc_math_util.h"
#include "mjpc/tasks/lsqp/lsqp.h"
#include "mjpc/planners/lsqp/lsqp_solver.h"
#include "mjpc/planners/lsqp/lsqp_collision_limit.h"

namespace mjpc {
void LsqpPlanner::Initialize(mjModel* model, const Task& task) {
  model_ = model;
  action_dim_ = model->nu;

  // Init task fabrics
  if (task.IsFabricsSupported()) {
    InitTaskFabrics();
  }

  // Init task Lsqp
  if (task.IsLSQPSupported()) {
    lsqp_task_ = dynamic_cast<Lsqp*>(const_cast<Task*>(&task));
    InitTaskLsqp(model_, lsqp_task_->data_);

    // Init main solver
    lsqp_solver_ = std::make_shared<LsqpSolver>(model_, lsqp_task_, owner_type_);
    lsqp_solver_->Init(lsqp_task_->data_, action_dim_);
  }

  // Init [cem_delegate_]
  if (cem_delegate_) {
    if (task.IsLSQPSupported()) {
      // Init [cem_delegate_]'s solvers
      cem_delegate_->SetPostResizeMjData([this, model]() {
        const auto& data_list = cem_delegate_->RolloutData();
        auto& solver_list = cem_delegate_->LsqpSolvers();
        const auto new_size = data_list.size();
        solver_list.reserve(new_size);
        while (solver_list.size() < new_size) {
          auto solver = std::make_shared<LsqpSolver>(model, lsqp_task_, MjOwnerAppType::MJPC);
          solver->Init(data_list[solver_list.size()].get(), action_dim_);
          solver_list.emplace_back(std::move(solver));
        }
      });

      // Lsqp control callback -> embedded into [cem_delegate_]
      // Using bound lambda instead of std::bind for clarity & a bit faster
      cem_delegate_->SetControlCallback(
          [this](double* policy_action, mjData* data, const BaseSolverPtr& solver) {
            return this->LsqpControl(policy_action, data,
                                     (solver != nullptr)
                                       ? std::dynamic_pointer_cast<LsqpSolver>(solver)
                                       : nullptr);
          });
    }

    // Init [cem_delegate_], which may invoke above callbacks
    cem_delegate_->Initialize(model, task);

    // Overwrite [cem_delegate_]'s actions dim & limits
    cem_delegate_->SetActionDim(CEM_PARAMS_TOTAL_DIM);
    std::vector<double> limits;
    for (auto i = 0; i < CEM_PARAMS_TOTAL_DIM; ++i) {
      limits.push_back(CEM_PARAMS_LIMIT_LOWER);
      limits.push_back(CEM_PARAMS_LIMIT_UPPER);
    }
    cem_delegate_->SetActionLimits(std::move(limits));
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
std::vector<double>
LsqpPlanner::LsqpControl(double* policy_action, mjData* data, const LsqpSolverPtr& solver) {
  if (((MjOwnerAppType::MJPC == owner_type_) && !planning_on_) || (nullptr == lsqp_task_)) {
    return std::vector(action_dim_, 0.);
  }

  // Use [lsqp_task_]'s data if not running in worker thread in MPC rollouts
  const bool rolllout_threaded_data = (nullptr != data);
  if (!rolllout_threaded_data) {
    data = lsqp_task_->data_;
  }

  // NOTE: THIS MUST TAKE INTO ACCOUNT OF BOTH SCENARIOS WHEREBY TARGET-OBJ STAYS ON GROUND & ALREADY IN-HAND
  // Current target-obj pos
  double target_obj_pos[3];
  mju_copy3(target_obj_pos, mjpc::QueryBodyPos(model_, data, Lsqp::TARGET_OBJ_NAME));

  // Current dist from [ee_target] -> [target_obj]
  const auto dist_to_target_obj = mju_dist3(target_obj_pos,
                                            mjpc::QuerySitePos(model_, data,
                                                               lsqp_task_->EETargetSiteName().data()));
  const bool has_reached_target_obj = (dist_to_target_obj <= 0.1);

  // APPLY [policy_action], if outputted from the delegate planner (eg: [cem_delegate_])
  if (policy_action) {
    double target_goal_pos[3];
    mju_copy3(target_goal_pos, mjpc::QuerySitePos(model_, data, Lsqp::TARGET_OBJ_GOAL_NAME));

    double target_delta_pos[3];
    mju_sub3(target_delta_pos, target_goal_pos, target_obj_pos);
    mju_normalize3(target_delta_pos);

    // [EE-mocap pos] perturbation along the path from [target_obj] -> [target_goal]
    const auto eetarget_mocap_name = lsqp_task_->EETargetMocapName();
    mjtNum new_ee_target_pos[3];
    mju_add3(new_ee_target_pos, mjpc::QueryBodyPos(model_, data, mjpc::Lsqp::TARGET_OBJ_NAME),
             (double [3]){0, 0, 0.07 + 0.1 * policy_action[0]}); // + 0.1 * policy_action[0]
    mjpc::SetBodyMocapPos(model_, data, eetarget_mocap_name.data(), new_ee_target_pos);
    if (!rolllout_threaded_data) {
      const MjpcSharedMutexLock lock(policy_mutex_);
      mju_copy3(visual_policy_ee_target_pos_, new_ee_target_pos);
    }

    // [EE-mocap quat]
#if 0
      // Rot around Y 90
      mjtNum new_ee_target_quat[4];
      mju_axisAngle2Quat(new_ee_target_quat, mjpc::UNIT_Y, M_PI_2);
      mjtNum delta_ee_target_quat_Z[4];
      mju_axisAngle2Quat(delta_ee_target_quat_Z, mjpc::UNIT_Z,
                         M_PI * (1 + policy_action[1]));
      //mjtNum delta_ee_target_quat_X[4];
      //mju_axisAngle2Quat(delta_ee_target_quat_X, mjpc::UNIT_X,
      //                  M_PI * (1 + policy_action[1]));
      mju_mulQuat(new_ee_target_quat, new_ee_target_quat, delta_ee_target_quat_Z);
      //mju_mulQuat(new_ee_target_quat, new_ee_target_quat, delta_ee_target_quat_X);
      mjpc::SetBodyMocapQuat(model_, data, lsqp_task_->EETargetMocapName().data(), new_ee_target_quat);
#else
    mjpc::SetBodyMocapQuat(model_, data, eetarget_mocap_name.data(), mjpc::Lsqp::EE_TARGET_PREGRASP_QUAT);
#endif
  } else {
    mju_rotVecQuat(palm_normal_, mjpc::UNIT_X,
                   mjpc::QuerySiteQuat(model_, data, lsqp_task_->EETargetSiteName().data()));
    mju_sub3(visual_ee_direction_, target_obj_pos,
             mjpc::QuerySitePos(model_, data, lsqp_task_->EETargetSiteName().data()));
    mju_normalize3(visual_ee_direction_);

    // Rotate [ee_target_mocap] once having reach [target_obj]
    if (has_reached_target_obj) {
      mjtNum new_ee_target_mocap_quat[3];
      mjpc::mjpc_quatFromVectors(new_ee_target_mocap_quat, visual_ee_direction_, mjpc::UNIT_X);
      mjpc::SetBodyMocapQuat(model_, data, lsqp_task_->EETargetMocapName().data(),
                             new_ee_target_mocap_quat);
    }
  }

  // [Fingertip-mocaps] perturbation
#if MJPC_LSQP_FINGERS_OSC
  if (has_reached_target_obj) {
    //uint8_t i = 0;
    for (const auto& fingertip : Lsqp::FINGERTIP_NAMES) {
      if (false) {
        //mjtNum new_finger_target_pos[3];
        //mju_addScl3(new_finger_target_pos, ee_target_pos,
        //            lsqp_task_->initial_fingertips_direction[fingertip],
        //           0.1 * policy_action[EE_CEM_PARAMS_DIM + (++i)]);
      } else {
        mjpc::SetBodyMocapPos(model_, data, lsqp_task_->FingertipTargetMocapName(fingertip).c_str(),
                              target_obj_pos);
      }
    }
  }
#endif

  // [Lsqp solver]: solve diff-ik
  // NOTE: This is made instance created per [LsqpControl()] to avoid dynamic allocation in threads,
  // which would cause sporadic crashes.
  std::vector<double> ctrl;
  if (rolllout_threaded_data) {
    assert(solver);
    ctrl = solver->Solve(data);
  } else {
    ctrl = lsqp_solver_->Solve(data);
  }
#if !MJPC_LSQP_FINGERS_OSC
    for (uint8_t i = IIWA14_DOF; i < IIWA14_DOF + ALLEGRO_DOF; ++i) {
      const int jnt_id = model_->dof_jntid[i];
      const double low_lim = model_->jnt_range[2 * jnt_id];
      const double high_lim = model_->jnt_range[2 * jnt_id + 1];
      ctrl[i] = low_lim + (policy_action
                             ? std::abs(policy_action[EE_CEM_PARAMS_DIM + (i - IIWA14_DOF)])
                             : mjpc::Random::rand()) * (high_lim - low_lim);
    }
#endif

  if ((MjOwnerAppType::MJAPP == owner_type_)
    /* &&lsqp_solver.config().CheckJointValues(ctrl.data(), ctrl.size(), 0.1)*/) {
    mju_copy(data->ctrl, ctrl.data(), ctrl.size());
  }
  mjpc::print(ctrl);
  return ctrl;
}

void LsqpPlanner::Traces(mjvScene* scn) {
  if (cem_delegate_) {
    cem_delegate_->Traces(scn);
  }
  if (!lsqp_task_) {
    return;
  }

  // Draw direction from [visual_policy_ee_target_pos_] -> [target_obj]
  static constexpr float GREEN[] = {0.0, 1.0, 0.0, 1.0};
  AddConnector(scn ? scn : lsqp_task_->scene_, mjGEOM_ARROW, 0.005,
               visual_policy_ee_target_pos_,
               mjpc::QueryBodyPos(model_, lsqp_task_->data_, Lsqp::TARGET_OBJ_NAME),
               GREEN);

  // Draw [visual_ee_direction_] from [ee_target_mocap]
  static constexpr float BLUE[] = {0.0, 0.0, 1.0, 1.0};
  const auto* ee_mocap_pos = mjpc::QueryBodyMocapPos(model_, lsqp_task_->data_,
                                                     lsqp_task_->EETargetMocapName().data());
  mjtNum ee_normal_end_pos[3];
  mju_addScl3(ee_normal_end_pos, ee_mocap_pos, visual_ee_direction_, 0.2);
  AddConnector(scn ? scn : lsqp_task_->scene_, mjGEOM_ARROW, 0.005,
               ee_mocap_pos,
               ee_normal_end_pos,
               BLUE);

  static constexpr float RED[] = {1.0, 0.0, 0.0, 1.0};
  const auto* ee_site_pos = mjpc::QuerySitePos(model_, lsqp_task_->data_,
                                               lsqp_task_->EETargetSiteName().data());
  mjtNum palm_normal_end_pos[3];
  mju_addScl3(palm_normal_end_pos, ee_site_pos, palm_normal_, 0.2);
  AddConnector(scn ? scn : lsqp_task_->scene_, mjGEOM_ARROW, 0.005,
               ee_site_pos,
               palm_normal_end_pos,
               RED);
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
    mju_copy(action, LsqpControl(action).data(), action_dim_);
  }
}
}
