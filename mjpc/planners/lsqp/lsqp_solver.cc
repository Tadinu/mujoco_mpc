#include "mjpc/planners/lsqp/lsqp_solver.h"

// mjpc
#include "mjpc/core/mjpc_common.h"
#include "mjpc/tasks/lsqp/lsqp.h"


namespace mjpc {
void LsqpSolver::Init(const mjData* data, int ndofs) {
  // 1- Create config
  config_ = LsqpConfig(model_, ndofs);

#if MJPC_LSQP_PLANAR_ROBOT
  // 2- Tasks
  // 2.1- End-effector task
  end_effector_subtask_ = LsqpFrameTask("EE", model_,
                                     "hand",
                                     mjOBJ_SITE,
                                     /*position_cost*/Eigen::VectorXd::Constant(1, 1.0),
                                     /*orientation_cost*/Eigen::VectorXd::Constant(1, 1.0),
                                     /*gain */ 1.0,
                                     /*lm_damping*/1.0);
  subtasks_ = std::vector<LsqpBaseTask*>{&end_effector_subtask_};

  // 3- Config limits
  config_limits_ = {std::make_shared<LsqpVelocityLimit>(model_, config_.ndofs(),
                                                        std::map<std::string, std::vector<double>>{
                                                            {"joint1", {M_PI}},
                                                            {"joint2", {M_PI}}}),
                    std::make_shared<LsqpCollisionLimit>(model_, config_.ndofs(), LsqpCollisionPairList{
                                                             {mjpc::GetBodyGeomIds(model_, "hand"),
                                                              {"floor", "obstacle_0", "obstacle_1"}}})
  };
#else
  // 2- Tasks
  // 2.1- End-effector task
  end_effector_subtask_ = LsqpFrameTask("EE", model_,
                                        lsqp_task_->EETargetSiteName(),
                                        mjOBJ_SITE,
                                        /*position_cost*/Eigen::VectorXd::Constant(1, 1.0),
                                        /*orientation_cost*/Eigen::VectorXd::Constant(1, 1.0),
                                        /*gain*/ 1.0,
                                        /*lm_damping*/1.0);

  // 2.2- Posture task (as biased pose in diff-ik solving, with cost being smaller than other tasks)
  posture_subtask_ = LsqpPostureTask("Posture", model_, /*cost*/Eigen::VectorXd::Constant(1, 5e-2));
  posture_subtask_.SetTarget(PosToEigen(lsqp_task_->system_qpos_home.data(), config_.nq()));

  // 2.3- Finger tasks
  const std::string attach_prefix = lsqp_task_->AttachmentPrefix();
  for (const auto& finger_tip : Lsqp::FINGERTIP_NAMES) {
    // LsqpRelativeFrameTask
    finger_subtasks_.emplace_back(/*task_name*/attach_prefix + finger_tip + "_" + Lsqp::PALM_NAME, model_,
                                               /*frame_name*/lsqp_task_->FingertipSiteName(finger_tip),
                                               /*frame_type*/mjOBJ_SITE,
                                               /*base_name*/lsqp_task_->PalmBodyName(),
                                               /*base_type*/mjOBJ_BODY,
                                               /*position_cost*/Eigen::VectorXd::Constant(1, 1.0),
                                               /*orientation_cost*/Eigen::VectorXd::Constant(1, 0.0),
                                               /*gain*/1.0,
                                               /*lm_damping*/1.0);
  }
  subtasks_ = std::vector<LsqpBaseTask*>{&end_effector_subtask_, &posture_subtask_};
  for (auto& finger_task : finger_subtasks_) {
    subtasks_.push_back(&finger_task);
  }

  // 3- Config limits
  config_limits_ = {std::make_shared<LsqpPositionLimit>(model_, config_.ndofs())};
  T_ee_initial_ = config_.GetTransformFrameToWorld(data, lsqp_task_->EETargetSiteName(), mjOBJ_SITE);
#endif
}

void LsqpSolver::SetFrameTaskTarget(mjData* data, LsqpFrameTask* task, const char* target_mocap_name) const {
  const auto* target_mocap_quat = mjpc::QueryBodyMocapQuat(model_, data, target_mocap_name);
  const auto* target_mocap_pos = mjpc::QueryBodyMocapPos(model_, data, target_mocap_name);
  const auto T_wt = LsqpSE3(target_mocap_quat, target_mocap_pos);
  task->SetTarget(T_wt);
}

std::vector<double> LsqpSolver::Solve(mjData* data) {
  // Update [end-effector task]'s mocap target
#if MJPC_LSQP_PLANAR_ROBOT
  SetFrameTaskTarget(data, &end_effector_subtask_, "target_mocap");
#else
  SetFrameTaskTarget(data, &end_effector_subtask_, lsqp_task_->EETargetMocapName().data());

  // Update [finger tasks]' targets & mocaps
  const auto attach_prefix = lsqp_task_->AttachmentPrefix();
  LsqpSE3 T_ee;
  const LsqpSE3 T_ee_prev = mju_isZero(data->userdata, model_->nuserdata)
                              ? T_ee_initial_
                              : LsqpSE3(data->userdata);
  for (auto i = 0; i < Lsqp::FINGERTIP_NAMES.size(); ++i) {
    T_ee = config_.GetTransformFrameToWorld(data, lsqp_task_->EETargetSiteName(), mjOBJ_SITE);
    const auto& fingertip = Lsqp::FINGERTIP_NAMES[i];

    // Update [finger_task]s target, as transform of fingertip mocaps relative to hand base
    const auto finger_target = lsqp_task_->FingertipTargetMocapName(fingertip);
    auto& finger_task = finger_subtasks_[i];
    const LsqpSE3 T_pm = config_.GetTransform(data, finger_target, mjOBJ_BODY,
                                              lsqp_task_->PalmBodyName(), mjOBJ_BODY);
    finger_task.SetTarget(T_pm);

    // Move fingertip mocaps to latest fingertips' poses
    const mjtNum* finger_target_quat = mjpc::QueryBodyMocapQuat(model_, data, finger_target.c_str());
    const mjtNum* finger_target_pos = mjpc::QueryBodyMocapPos(model_, data, finger_target.c_str());
    const LsqpSE3 dT = T_ee * T_ee_prev.Inverse();
    const auto T_w_mocap = LsqpSE3(finger_target_quat, finger_target_pos);
    const LsqpSE3 T_w_mocap_new = dT * T_w_mocap;
    mjpc::SetBodyMocapPos(model_, data, finger_target.c_str(),
                          T_w_mocap_new.Translation().data());
    mjpc::SetBodyMocapQuat(model_, data, finger_target.c_str(),
                           T_w_mocap_new.Rotation().Wxyz().data());
  }
#endif

  // Init [ctrl] as current [data->qpos]
  auto ctrl = std::vector<double>(data->qpos, data->qpos + config_.ndofs());

  // Compute velocity and integrate into the next configuration
  const Eigen::VectorXd vel = IK_Solve(data, config_, subtasks_, INTEGRATION_DT, /*damping*/1e-3,
                                       config_limits_);
#if MJPC_LSQP_PLANAR_ROBOT
  if (false == (vel.isZero() || vel.hasNaN())) {
    mju_addToScl(ctrl.data(), vel.data(), INTEGRATION_DT, ctrl.size());
    mju_copy(data->ctrl, ctrl.data(), ctrl.size());
  }
  else {
    mju_zero(ctrl.data(), ctrl.size());
  }
#else
  if (!vel.hasNaN()) {
    if constexpr (Lsqp::POSITION_CTRL_ENABLED) {
      // Integrate [vel] into current [q]
      // NOTE: DO NOT USE mj_integratePos() here due to its looping over all joints of not only robots but also objects
      mju_addToScl(ctrl.data(), vel.data(), INTEGRATION_DT, ctrl.size());
    } else {
      // [vel] -> [ctrl]
      mju_copy(ctrl.data(), vel.data(), ctrl.size());
    }
  } else {
    mju_zero(ctrl.data(), ctrl.size());
  }
#endif

  // Save latest [T_ee] to [data->userdata]
  mju_copy(data->userdata, T_ee.Parameters().data(), T_ee.PARAMS_DIM);
  return ctrl;
}
} // end namespace mjpc