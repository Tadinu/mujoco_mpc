#include "mjpc/planners/lsqp/lsqp_planner.h"
#include "mjpc/tasks/lsqp/lsqp.h"

namespace mjpc {
std::vector<std::string> LsqpPlanner::FingertipNames() const {
  return lsqp_task_ ? lsqp_task_->FingertipNames() : std::vector<std::string>();
}

std::map<std::string, std::vector<float>> LsqpPlanner::FingertipRGBAs() const {
  return lsqp_task_ ? lsqp_task_->FingertipRGBAs() : std::map<std::string, std::vector<float>>();
}

void LsqpPlanner::Initialize(mjModel* model, const Task& task) {
  lsqp_task_ = dynamic_cast<Lsqp*>(const_cast<Task*>(&task));
  model_ = model;
  data_ = lsqp_task_ ? lsqp_task_->data_ : nullptr;

  // dimensions
  dim_state_ = model->nq + model->nv + model->na; // state dimension
  dim_state_derivative_ = 2 * model->nv + model->na; // state derivative dimension
  dim_action_ = task.GetActionDim(); // action dimension
  dim_sensor_ = model->nsensordata; // number of sensor values
  dim_max_ = std::max({dim_state_, dim_state_derivative_, dim_action_, model->nuser_sensor});

  if (trajectory_) {
    trajectory_->Reset(0);
  } else {
    trajectory_ = std::make_shared<Trajectory>();
  }

  // Init task fabrics
  if (task.IsFabricsSupported()) {
    InitTaskFabrics();
  }

  // Init Lsqp
  if (task.IsLSQPSupported()) {
    InitLsqpEnv(model_, data_);
  }
}

void LsqpPlanner::InitLsqpEnv(const mjModel* model, mjData* data) {
  // 1- Create config
  config_ = LsqpConfig(model, data);

#if MJPC_LSQP_PLANAR_ROBOT
  // 2- Tasks
  // 2.1- End-effector task
  end_effector_task_ = LsqpFrameTask("EE", model,
                                     "hand",
                                     mjOBJ_SITE,
                                     /*position_cost*/Eigen::VectorXd::Constant(1, 1.0),
                                     /*orientation_cost*/Eigen::VectorXd::Constant(1, 1.0),
                                     /*gain */ 1.0,
                                     /*lm_damping*/1.0);
  subtasks_ = std::vector<LsqpBaseTask*>{&end_effector_task_};

  // 3- Config limits
  config_limits_ = {std::make_shared<LsqpVelocityLimit>(model, std::map<std::string, std::vector<double>>{
                                                            {"joint1", {M_PI}},
                                                            {"joint2", {M_PI}}})};
#else
  // 2- Tasks
  // 2.1- End-effector task
  end_effector_task_ = LsqpFrameTask("EE", model,
                                     "attachment_site",
                                     mjOBJ_SITE,
                                     /*position_cost*/Eigen::VectorXd::Constant(1, 1.0),
                                     /*orientation_cost*/Eigen::VectorXd::Constant(1, 1.0),
                                     /*gain*/ 1.0,
                                     /*lm_damping*/1.0);

  // 2.2- Posture task
  posture_task_ = LsqpPostureTask("Posture", model, /*cost*/Eigen::VectorXd::Constant(1, 5e-2));
  posture_task_.SetTarget(PosToEigen(data->qpos, model->nq));

  // 2.3- Finger tasks
  const std::string attach_prefix = GetAttachmentPrefix();
  for (const auto& finger_tip : FingertipNames()) {
    // LsqpRelativeFrameTask
    finger_tasks_.emplace_back(attach_prefix + finger_tip + "_palm", model,
                               /*frame_name*/attach_prefix + finger_tip,
                               /*frame_type*/mjOBJ_SITE,
                               /*base_name*/attach_prefix + "palm",
                               /*base_type*/mjOBJ_BODY,
                               /*position_cost*/Eigen::VectorXd::Constant(1, 1.0),
                               /*orientation_cost*/Eigen::VectorXd::Constant(1, 0.0),
                               /*gain*/1.0,
                               /*lm_damping*/1.0);
  }
  subtasks_ = std::vector<LsqpBaseTask*>{&end_effector_task_, &posture_task_};
  for (auto& finger_task : finger_tasks_) {
    subtasks_.push_back(&finger_task);
  }

  // 3- Config limits
  config_limits_ = {std::make_shared<LsqpPositionLimit>(model)};
  T_ee_prev_ = config_.GetTransformFrameToWorld("attachment_site", mjOBJ_SITE);
#endif

  // 4- Init mocaps
  if (lsqp_task_) {
    // In [mjapp], [task_] may not have already been initialized with [model, data]
    lsqp_task_->model_ = const_cast<mjModel*>(model);
    lsqp_task_->data_ = data;
    lsqp_task_->InitMocaps();
  }
}

void LsqpPlanner::RefreshMj(mjModel* model, mjData* data) {
  if (MjOwnerAppType::MJPC == owner_type_) {
    config_.RefreshMj(model, data);
    for (auto& cl : config_limits_) {
      cl->RefreshMj(model);
    }
  }
}

std::string LsqpPlanner::GetAttachmentPrefix() const {
  return lsqp_task_ ? lsqp_task_->AttachmentPrefix() : "";
}

void LsqpPlanner::Allocate() {
  trajectory_->Initialize(dim_state_, dim_action_, lsqp_task_ ? lsqp_task_->num_residual : 1,
                          lsqp_task_ ? lsqp_task_->num_trace : 1, 1);
  trajectory_->Allocate(1);
}


void LsqpPlanner::SetFrameTaskTarget(LsqpFrameTask* task, const char* target_mocap_name) const {
  const auto* target_mocap_quat = lsqp_task_->QueryBodyMocapQuat(target_mocap_name);
  const auto* target_mocap_pos = lsqp_task_->QueryBodyMocapPos(target_mocap_name);
  const auto T_wt = LsqpSE3(target_mocap_quat, target_mocap_pos);
  task->SetTarget(T_wt);
}

void LsqpPlanner::LsqpControl(bool position_ctrl) {
  auto* model = lsqp_task_->model_;
  auto* data = lsqp_task_->data_;
  const bool current_config_valid = config_.CheckJointLimits();
  if (false == current_config_valid) {
    mju_zero(data->ctrl, IIWA14_DOF);
    mju_zero(data->qvel + IIWA14_DOF, ALLEGRO_DOF);
    lsqp_task_->InitMocaps();
    return;
  }

#if MJPC_LSQP_PLANAR_ROBOT
  SetFrameTaskTarget(&end_effector_task_, "target_mocap");
#else
  // Update kuka end-effector task
  SetFrameTaskTarget(&end_effector_task_, "target");

  // Update finger tasks' targets & mocaps
  static const auto fingertip_names = FingertipNames();
  static const auto attach_prefix = GetAttachmentPrefix();
  LsqpSE3 T_ee;
  for (auto i = 0; i < fingertip_names.size(); ++i) {
    T_ee = config_.GetTransformFrameToWorld("attachment_site", mjOBJ_SITE);
    const auto& fingertip = fingertip_names[i];

    // Update [finger_task]s target, as transform of fingertip mocaps relative to hand base
    const auto finger_target = attach_prefix + fingertip + "_target";
    auto& finger_task = finger_tasks_[i];
    const LsqpSE3 T_pm = config_.GetTransform(finger_target, mjOBJ_BODY,
                                              attach_prefix + "palm", mjOBJ_BODY);
    finger_task.SetTarget(T_pm);

    // Move fingertip mocaps to latest fingertips' poses
    const mjtNum* finger_target_quat =
        lsqp_task_->QueryBodyMocapQuat(finger_target.c_str());
    const mjtNum* finger_target_pos = lsqp_task_->QueryBodyMocapPos(finger_target.c_str());
    const LsqpSE3 dT = T_ee * T_ee_prev_.Inverse();
    const auto T_w_mocap = LsqpSE3(finger_target_quat, finger_target_pos);
    const LsqpSE3 T_w_mocap_new = dT * T_w_mocap;
    lsqp_task_->SetBodyMocapPos(finger_target.c_str(),
                                T_w_mocap_new.Translation().data());
    lsqp_task_->SetBodyMocapQuat(finger_target.c_str(),
                                 T_w_mocap_new.Rotation().Wxyz().data());
  }
#endif

  // Get current [q] from [data->qpos]
  Eigen::VectorXd q(model->nq);
  mju_copy(q.data(), data->qpos, q.size());

#if MJPC_LSQP_PLANAR_ROBOT
  // Compute velocity and integrate into the next configuration
  static float INTEGRATION_DT = 0.001;
  for (auto i = 0; i < 20; ++i) {
    const Eigen::VectorXd vel = IK_Solve(config_, subtasks_, INTEGRATION_DT, /*damping*/1e-3, config_limits_);
    if (false == (vel.isZero() || vel.hasNaN())) {
      mj_integratePos(model, q.data(), vel.data(), INTEGRATION_DT);
      const Eigen::VectorXd err = end_effector_task_.ComputeError(config_);
      if ((err.head<3>().norm() <= 1e-4) && (err.tail<3>().norm() <= 1e-4)) {
        break;
      }
    }
  }
  mju_copy(data->ctrl, q.data(), q.size());
#else
  // Compute velocity and integrate into the next configuration
  static float INTEGRATION_DT = 0.01;
  const Eigen::VectorXd vel = IK_Solve(config_, subtasks_, INTEGRATION_DT, /*damping*/1e-3, config_limits_);

  if (!vel.hasNaN()) {
    // Integrate [vel] into current [q]
    mj_integratePos(model, q.data(), vel.data(), INTEGRATION_DT);

    const bool valid_q = !q.hasNaN() && config_.CheckJointValues(q);
    if (valid_q) {
      if (MjOwnerAppType::MJAPP == owner_type_) {
        // MJAPP
        if (position_ctrl) {
          mju_copy(data->qpos, q.data(), q.size());
        } else {
          mju_copy(data->ctrl, vel.data(), IIWA14_DOF);
          mju_copy(data->qpos + IIWA14_DOF, q.data() + IIWA14_DOF, ALLEGRO_DOF);
        }
      } else {
        // MJPC
        const MjpcSharedMutexLock lock(policy_mutex_);
        action_.resize(q.size());
        mju_copy(action_.data(), vel.data(), IIWA14_DOF);
        mju_copy(action_.data() + IIWA14_DOF, q.data() + IIWA14_DOF, ALLEGRO_DOF);
      }
    }
  }

  //mj_camlight(model, data);

  // Save latest [T_ee] to [T_ee_prev]
  T_ee_prev_ = T_ee;
#endif
}

void LsqpPlanner::Traces(mjvScene* scn) {
#if 0
  std::vector<double> traces;
  {
    const MjpcSharedMutexLock lock(policy_mutex_);
    if (trajectory_->trace.size() >= 6) {
      traces = trajectory_->trace;
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

void LsqpPlanner::ActionFromPolicy(double* action, const double* state, double time, bool use_previous) {
  const MjpcSharedMutexLock lock(policy_mutex_);

  // WAIT FOR ACTION TO BE COMPUTED
  if (action_.empty()) {
    return;
  }

  // APPLY ACTION: COPY [action_] -> [action]
#if 0
  const mjtNum* target_pos = task_->QueryTargetPos();
  if (target_pos) {
    trajectory_->trace.push_back(target_pos[0]);
    trajectory_->trace.push_back(target_pos[1]);
    trajectory_->trace.push_back(target_pos[2]);
  }
#endif
  print("ACTION", action_);
  mju_copy(action, action_.data(), action_.size());

  // Clear [action_]
  action_.clear();

  // Clamp controls on outputted [action]
  Clamp(action, model_->actuator_ctrlrange, model_->nu);
}
}
