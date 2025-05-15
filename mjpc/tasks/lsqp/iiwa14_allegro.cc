#include "mjpc/tasks/lsqp/iiwa14_allegro.h"

#include <string>

// mjpc
#include "mjpc/planners/lsqp/lsqp_planner.h"

namespace mjpc {
const std::vector<std::string> IIWA14Allegro::FINGERTIP_NAMES = {"rf_tip", "mf_tip", "ff_tip", "th_tip"};
const std::map<std::string, std::vector<float>> IIWA14Allegro::FINGERTIPS_RGBA = {
    {FINGERTIP_NAMES[0], {0.9f, 0.f, 0.f, 1.f}}, // Red
    {FINGERTIP_NAMES[1], {0.f, 0.9f, 0.f, 1.f}}, // Green
    {FINGERTIP_NAMES[2], {0.f, 0.f, 0.9f, 1.f}}, // Blue
    {FINGERTIP_NAMES[3], {0.9f, 0.9f, 0.9f, 1.f}}}; // White
const std::vector<const char*> IIWA14Allegro::FINGERS_JOINT_NAMES = {
    "rfj0", "rfj1", "rfj2", "rfj3",
    "mfj0", "mfj1", "mfj2", "mfj3",
    "ffj0", "ffj1", "ffj2", "ffj3",
    "thj0", "thj1", "thj2", "thj3"
};

const std::vector<const char*> IIWA14Allegro::FINGERS_ACTUATOR_NAMES = {
    "ffa0", "ffa1", "ffa2", "ffa3",
    "mfa0", "mfa1", "mfa2", "mfa3",
    "rfa0", "rfa1", "rfa2", "rfa3",
    "tha0", "tha1", "tha2", "tha3"
};

void IIWA14Allegro::InitSolverConfigs(const LsqpSolverPtr& solver, const mjData* data,
                                      int ndofs) {
  // 1- Create config
  solver->config_ = LsqpConfig(model_, ndofs);

#if MJPC_LSQP_PLANAR_ROBOT
  // 2- Tasks
  // 2.1- End-effector task
  solver->end_effector_subtasks_.emplace_back(LsqpRelativeFrameTask{"EE", model_,
                                                                    "hand",
                                                                    mjOBJ_SITE,
                                                                    "link0",
                                                                    mjOBJ_BODY,
                                                                    /*position_cost*/
                                                                    Eigen::VectorXd::Constant(1, 1.0),
                                                                    /*orientation_cost*/
                                                                    Eigen::VectorXd::Constant(1, 1.0),
                                                                    /*gain */ 1.0,
                                                                    /*lm_damping*/1.0});
  solver->subtasks_ = std::vector<LsqpBaseTask*>{&solver->end_effector_subtasks_[0]};

  // 3- Config limits
  solver->config_limits_ = {std::make_shared<LsqpVelocityLimit>(model_, solver->config_.ndofs(),
                                                                std::map<std::string, std::vector<double>>{
                                                                    {"joint1", {M_PI}},
                                                                    {"joint2", {M_PI}}}),
                            std::make_shared<LsqpCollisionLimit>(model_, solver->config_.ndofs(),
                                                                 LsqpCollisionPairList{
                                                                     {mjpc::GetBodyGeomIds(model_, "hand"),
                                                                      {"floor", "obstacle_0", "obstacle_1"}}})
  };
#else
  // 2- Tasks
  // 2.1- End-effector task
  solver->end_effector_subtasks_.emplace_back(LsqpRelativeFrameTask{"EE", model_,
                                                                    EETargetSiteName(),
                                                                    mjOBJ_SITE,
                                                                    BASE_LINK_NAME,
                                                                    mjOBJ_BODY,
                                                                    /*position_cost*/
                                                                    Eigen::VectorXd::Constant(1, 1.0),
                                                                    /*orientation_cost*/
                                                                    Eigen::VectorXd::Constant(1, 1.0),
                                                                    /*gain*/ 1.0,
                                                                    /*lm_damping*/1.0
  });

  // 2.2- Posture task (as biased pose in diff-ik solving, with cost being smaller than other tasks)
  solver->posture_subtask_ = LsqpPostureTask("Posture", model_, /*cost*/
                                             Eigen::VectorXd::Constant(1, 5e-2));
  solver->posture_subtask_.SetTarget(PosToEigen(system_qpos_home_.data(), solver->config_.nq()));

  // 2.3- Finger tasks
  const std::string attach_prefix = AttachmentPrefix();
  for (const auto& finger_tip : IIWA14Allegro::FINGERTIP_NAMES) {
    // LsqpRelativeFrameTask
    solver->finger_subtasks_.emplace_back(
        /*task_name*/attach_prefix + finger_tip + "_" + IIWA14Allegro::PALM_NAME, model_,
                     /*frame_name*/FingertipSiteName(finger_tip),
                     /*frame_type*/mjOBJ_SITE,
                     /*base_name*/PalmBodyName(),
                     /*base_type*/mjOBJ_BODY,
                     /*position_cost*/Eigen::VectorXd::Constant(1, 1.0),
                     /*orientation_cost*/Eigen::VectorXd::Zero(1),
                     /*gain*/1.0,
                     /*lm_damping*/1.0);
  }
  solver->subtasks_.push_back(&solver->posture_subtask_);
  for (auto& ee_task : solver->end_effector_subtasks_) {
    solver->subtasks_.push_back(&ee_task);
  }
  for (auto& finger_task : solver->finger_subtasks_) {
    solver->subtasks_.push_back(&finger_task);
  }

  // 3- Config limits
  solver->config_limits_ = {std::make_shared<LsqpPositionLimit>(model_, solver->config_.ndofs())};
  solver->T_wrist_initial_ = solver->config_.GetTransformFrameToWorld(
      data, ATTACHMENT_SITE_NAME, mjOBJ_SITE);
#endif
}

std::vector<double>
IIWA14Allegro::Control(double* policy_action, mjData* data, const LsqpSolverPtr& solver) {
  // Use task data if not running in worker thread in MPC rollouts
  const bool is_rollout_thread_data = (nullptr != data);
  if (!is_rollout_thread_data) {
    data = data_;
    if (!data_) {
      return {};
    }
  }

#if MJPC_LSQP_AUTO_MODE
  const auto eetarget_mocap_name = EETargetMocapName();
#if 0
  // RECORD EE_TARGET-POSE
  double delta_pos[3];
  mju_sub3(delta_pos, mjpc::QueryBodyPos(model, data, mjpc::Lsqp::TARGET_OBJ_NAME),
           mjpc::QueryBodyMocapPos(model, data, eetarget_mocap_name.data()));
  mjpc::print("Pos", delta_pos[0], delta_pos[1], delta_pos[2]);

  double quat[4];
  mju_copy4(quat, mjpc::QueryBodyMocapQuat(model, data, eetarget_mocap_name.data()));
  mjpc::print("Quat", quat[0], quat[1], quat[2], quat[3]);
#endif

  // MOVE [ee_target] MOCAP TO A SPECIFIC EE_TARGET-POSE
  if (false) {
#if 1
    mjtNum target_pos[3];
    mju_add3(target_pos, mjpc::QueryBodyPos(model_, data, mjpc::Lsqp::TARGET_OBJ_NAME),
             (double [3]){0, 0, 0.07});
#else
    // MOVE [ee_target] MOCAP AROUND
    static constexpr float radius = 0.5;
    mjtNum target_pos[3] = {
      lsqp_task_->initial_ee_target_pos[0] /* + radius * cos(M_PI * data->time)*/,
      lsqp_task_->initial_ee_target_pos[1] + radius * sin(M_PI * data->time),
      lsqp_task_->initial_ee_target_pos[2]};
#endif
    mjpc::SetBodyMocapPos(model_, data, eetarget_mocap_name.data(), target_pos);
    mjpc::SetBodyMocapQuat(model_, data, eetarget_mocap_name.data(),
                           mjpc::IIWA14Allegro::EE_TARGET_PREGRASP_QUAT);
  }
#endif

  // NOTE: THIS MUST TAKE INTO ACCOUNT OF BOTH SCENARIOS WHEREBY TARGET-OBJ STAYS ON GROUND & ALREADY IN-HAND
  // Current target-obj pos
  double target_obj_pos[3];
  mju_copy3(target_obj_pos, mjpc::QueryBodyPos(model_, data, Lsqp::TARGET_OBJ_NAME));

  // Current dist from [ee_target] -> [target_obj]
  const auto dist_to_target_obj = mju_dist3(target_obj_pos,
                                            mjpc::QuerySitePos(model_, data,
                                                               EETargetSiteName().data()));
  const bool has_reached_target_obj = (dist_to_target_obj <= 0.2);

  // APPLY [policy_action], if outputted from the delegate planner (eg: [cem_delegate_])
  if (policy_action) {
    double target_goal_pos[3];
    mju_copy3(target_goal_pos, mjpc::QuerySitePos(model_, data, Lsqp::TARGET_OBJ_GOAL_NAME));

    double target_delta_pos[3];
    mju_sub3(target_delta_pos, target_goal_pos, target_obj_pos);
    mju_normalize3(target_delta_pos);

    // [EE-mocap pos] perturbation along the path from [target_obj] -> [target_goal]
    const auto eetarget_mocap_name = EETargetMocapName();
    mjtNum new_ee_target_pos[3];
    mju_add3(new_ee_target_pos, mjpc::QueryBodyPos(model_, data, mjpc::Lsqp::TARGET_OBJ_NAME),
             (double [3]){0.065, 0, 0.02 + 0.1 * std::abs(policy_action[0])}); // + 0.1 * policy_action[0]
    mjpc::SetBodyMocapPos(model_, data, eetarget_mocap_name.data(), new_ee_target_pos);
    if (!is_rollout_thread_data) {
      const MjpcMutexLock lock(mutex_);
      mju_copy3(visual_policy_ee_target_pos_, new_ee_target_pos);
    }

    // [EE-mocap quat]
#if (EE_CEM_PARAMS_DIM > 1)
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
    mjpc::SetBodyMocapQuat(model_, data, eetarget_mocap_name.data(), IIWA14Allegro::EE_TARGET_PREGRASP_QUAT);
#endif
  } else {
    mju_rotVecQuat(palm_normal_, mjpc::UNIT_X,
                   mjpc::QuerySiteQuat(model_, data, EETargetSiteName().data()));
    mju_sub3(visual_ee_direction_, target_obj_pos,
             mjpc::QuerySitePos(model_, data, EETargetSiteName().data()));
    mju_normalize3(visual_ee_direction_);

    // Rotate [ee_target_mocap] once having reach [target_obj]
    if (has_reached_target_obj) {
      mjtNum new_ee_target_mocap_quat[3];
      mjpc::MjuQuatFromVectors(new_ee_target_mocap_quat, visual_ee_direction_, mjpc::UNIT_X);
      mjpc::SetBodyMocapQuat(model_, data, EETargetMocapName().data(),
                             new_ee_target_mocap_quat);
    }
  }

  // [Fingertip-mocaps] perturbation
#if MJPC_LSQP_FINGERS_OSC
  if (has_reached_target_obj) {
    uint8_t i = 0;
    for (const auto& fingertip : IIWA14Allegro::FINGERTIP_NAMES) {
      if constexpr (FINGERS_CEM_PARAMS_DIM > 0) {
        mjtNum new_finger_target_pos[3];
        mju_addScl3(new_finger_target_pos, target_obj_pos,
                    initial_fingertips_direction[fingertip],
                    0.1 * policy_action[EE_CEM_PARAMS_DIM + (i++)]);
        mjpc::SetBodyMocapPos(model_, data, FingertipTargetMocapName(fingertip).c_str(),
                              new_finger_target_pos);
      } else {
        mjpc::SetBodyMocapPos(model_, data, FingertipTargetMocapName(fingertip).c_str(),
                              target_obj_pos);
      }
    }
  }
#endif

  // [Lsqp solver]: solve diff-ik
  std::vector<double> ctrl = is_rollout_thread_data ? Solve(solver, data) : Solve(lsqp_solver_, data);
  if (ctrl.size()) {
    if (is_rollout_thread_data && !MJPC_LSQP_FINGERS_OSC) {
      for (uint8_t i = IIWA14_DOF; i < IIWA14_DOF + ALLEGRO_DOF; ++i) {
        const int jnt_id = model_->dof_jntid[i];
        const double low_lim = model_->jnt_range[2 * jnt_id];
        const double high_lim = model_->jnt_range[2 * jnt_id + 1];
        ctrl[i] = low_lim + (policy_action
                               ? std::abs(policy_action[EE_CEM_PARAMS_DIM + (i - IIWA14_DOF)])
                               : mjpc::Random::rand()) * (high_lim - low_lim);
      }
      if (!has_reached_target_obj) {
        mju_zero(ctrl.data() + IIWA14_DOF, ALLEGRO_DOF);
      }
    }
  } else {
    ctrl = std::vector<double>(IIWA14_DOF + ALLEGRO_DOF, 0.0);
  }

  mjpc::print("[IIWA14Allegro] Solved ctrl", ctrl);
  return ctrl;
}

std::vector<double> IIWA14Allegro::Solve(const LsqpSolverPtr& solver, const mjData* data) {
  // Update [end-effector task]'s mocap target
#if MJPC_LSQP_PLANAR_ROBOT
  solver->SetFrameTaskTarget(data, &solver->end_effector_subtasks_[0], "target_mocap");
#else
  solver->SetFrameTaskTarget(data, &solver->end_effector_subtasks_[0], EETargetMocapName().data());

  // Update [finger tasks]' targets & mocaps
  const auto attach_prefix = AttachmentPrefix();
  LsqpSE3 T_wrist;
  const LsqpSE3 T_wrist_prev = mju_isZero(data->userdata, model_->nuserdata)
                                 ? solver->T_wrist_initial_
                                 : LsqpSE3(data->userdata);
  for (auto i = 0; i < IIWA14Allegro::FINGERTIP_NAMES.size(); ++i) {
    T_wrist = solver->config_.GetTransformFrameToWorld(data, ATTACHMENT_SITE_NAME, mjOBJ_SITE);
    const auto& fingertip = IIWA14Allegro::FINGERTIP_NAMES[i];

    // Update [finger_task]s target, as transform of fingertip mocaps relative to hand base
    const auto finger_target = FingertipTargetMocapName(fingertip);
    auto& finger_task = solver->finger_subtasks_[i];
    const LsqpSE3 T_pm = solver->config_.GetTransform(data, finger_target, mjOBJ_BODY,
                                                      PalmBodyName(), mjOBJ_BODY);
    finger_task.SetTarget(T_pm);

    // Move fingertip mocaps to latest fingertips' poses
    const mjtNum* finger_target_quat = mjpc::QueryBodyMocapQuat(model_, data, finger_target.c_str());
    const mjtNum* finger_target_pos = mjpc::QueryBodyMocapPos(model_, data, finger_target.c_str());
    const LsqpSE3 dT = T_wrist * T_wrist_prev.Inverse();
    const auto T_w_mocap = LsqpSE3(finger_target_quat, finger_target_pos);
    const LsqpSE3 T_w_mocap_new = dT * T_w_mocap;
    mjpc::SetBodyMocapPos(model_, data, finger_target.c_str(),
                          T_w_mocap_new.Translation().data());
    mjpc::SetBodyMocapQuat(model_, data, finger_target.c_str(),
                           T_w_mocap_new.Rotation().Wxyz().data());
  }
#endif

  // Compute velocity and integrate into the next configuration
  const Eigen::VectorXd vel = mjpc::IK_Solve(data, solver->config_, solver->subtasks_,
                                             INTEGRATION_DT, /*damping*/1e-3,
                                             solver->config_limits_);
  // NOTE: vel.size() == model->nv (qvel's size == num of dofs)
  std::vector<double> ctrl;
  if (!vel.hasNaN()) {
    if constexpr (Lsqp::POSITION_CTRL_ENABLED) {
      // Integrate [vel] into current [q]
      // Init [ctrl] as current [data->qpos] -> NOTE: Must always get the full qpos[nq]
      ctrl = std::vector<double>(data->qpos, data->qpos + model_->nq);
      mj_integratePos(model_, ctrl.data(), vel.data(), INTEGRATION_DT);
    } else {
      // [vel] -> [ctrl]
      ctrl = std::vector<double>(vel.size(), 0.0);
      mju_copy(ctrl.data(), vel.data(), ctrl.size());
    }
  }

  // Save latest [T_wrist] to [data->userdata]
  mju_copy(data->userdata, T_wrist.Parameters().data(), T_wrist.PARAMS_DIM);
  return ctrl;
}

void IIWA14Allegro::DrawTraces() {
  if (!scene_) {
    return;
  }
  // Draw direction from [visual_policy_ee_target_pos_] -> [target_obj]
  static constexpr float GREEN[] = {0.0, 1.0, 0.0, 1.0};
  AddConnector(scene_, mjGEOM_ARROW, 0.005,
               visual_policy_ee_target_pos_,
               mjpc::QueryBodyPos(model_, data_, Lsqp::TARGET_OBJ_NAME),
               GREEN);

  // Draw [visual_ee_direction_] from [ee_target_mocap]
  static constexpr float BLUE[] = {0.0, 0.0, 1.0, 1.0};
  const auto* ee_mocap_pos = mjpc::QueryBodyMocapPos(model_, data_,
                                                     EETargetMocapName().data());
  mjtNum ee_normal_end_pos[3];
  mju_addScl3(ee_normal_end_pos, ee_mocap_pos, visual_ee_direction_, 0.2);
  AddConnector(scene_, mjGEOM_ARROW, 0.005,
               ee_mocap_pos,
               ee_normal_end_pos,
               BLUE);

  static constexpr float RED[] = {1.0, 0.0, 0.0, 1.0};
  const auto* ee_site_pos = mjpc::QuerySitePos(model_, data_,
                                               EETargetSiteName().data());
  mjtNum palm_normal_end_pos[3];
  mju_addScl3(palm_normal_end_pos, ee_site_pos, palm_normal_, 0.2);
  AddConnector(scene_, mjGEOM_ARROW, 0.005,
               ee_site_pos,
               palm_normal_end_pos,
               RED);
}

void IIWA14Allegro::TransitionLocked(mjModel* model, mjData* data) {
  Task::TransitionLocked(model, data);
  double residuals[100];
  double terms[10];
  residual_.Residual(model, data, residuals);
  residual_.CostTerms(terms, residuals, /*weighted=*/false);
  //mjpc::print("WEIGHT:", weight[0], weight[1]);
  //mjpc::print("TERMS:", terms[0], terms[1]);

  // reach is solved:
  auto& norm_type = data->userdata[0];
  const auto& reach_distance = terms[0]; // Distance to target object
  if (data->time > 0 && norm_type == 0 && reach_distance < 0.04) {
    weight[0] = 0; // disable reach
    weight[1] = 1; // enable bring
    norm_type = 2;
  }

  // bring is solved, reset:
  const auto& bring_distance = terms[1]; // Distance to target goal
  if (norm_type == 2 && bring_distance < 0.01) {
    weight[0] = 1; // enable reach
    weight[1] = 0; // disable bring
    norm_type = 0;
  }

  // Init once only, already checked here-in
  if (lsqp_planner_ && IsLSQPSupported()) {
    lsqp_planner_->InitTaskLsqp(model, data);
  }

  // Reset target obj if being flung away
  if (mju_dist3(mjpc::QueryBodyPos(model_, data, Lsqp::TARGET_OBJ_NAME), (double[3]){0, 0, 0}) > 1) {
    int obj_id = mj_name2id(model, mjOBJ_BODY, TARGET_OBJ_NAME);
    if (obj_id != -1) {
      int jnt_qposadr = model->jnt_qposadr[model->body_jntadr[obj_id]];
      int jnt_veladr = model->jnt_dofadr[model->body_jntadr[obj_id]];
      mju_copy(data->qpos + jnt_qposadr, model->qpos0 + jnt_qposadr, mjpc::CArraySize(TARGET_OBJ_QPOS));
      mju_zero(data->qvel + jnt_veladr, 6);
    }
    mutex_.unlock(); // step calls sensor that calls Residual.
    mj_forward(model, data); // mj_step1 would suffice, we just need contact
    mutex_.lock();
  }
}

void IIWA14Allegro::ResidualFn::Residual(const mjModel* model, const mjData* data, double* residual) const {
  const Lsqp* lsqp_task = static_cast<const Lsqp*>(task_);
  int counter = 0;

  // Obj position, quat
  double* obj_pos = SensorByName(model, data, std::string(TARGET_OBJ_NAME) + "_pos");
  double* obj_quat = SensorByName(model, data, std::string(TARGET_OBJ_NAME) + "_quat");

  // ---------- Residual (0) ----------
  // EE target position
  double* ee_target_pos = SensorByName(model, data, lsqp_task->EETargetSiteName() + "_pos");

  // reach error
  mju_sub3(residual + counter, obj_pos, ee_target_pos);
  counter += 3;

#if IIWA14_ALLEGRO_BRING
  // ---------- Residual (1) ----------
  // goal position error
  mju_sub3(residual + counter, mjpc::QuerySitePos(model, data, TARGET_OBJ_GOAL_NAME), obj_pos);
  counter += 3;

  // goal orientation error
  mju_subQuat(residual + counter, mjpc::QuerySiteQuat(model, data, TARGET_OBJ_GOAL_NAME), obj_quat);
  counter += 4;

  // ---------- Residual (2) ----------
  // grasp error
  residual[counter++] = cost_calc_.TotalCost();
  //std::cout << "GRASP COST: " << residual[counter - 1] << std::endl;
#endif

  // Sanity check
  CheckSensorDim(model, counter);
}
}