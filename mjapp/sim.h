#pragma once

#include <atomic>
#include <chrono>
#include <utility>
#include <filesystem>

// Eigen
#include <Eigen/Core>

// MuJoCo
#include <mujoco/mujoco.h>
#include <array_safety.h>
#include <fmt/format.h>

// mjpc
#include "mjpc/sim_base.h"
#include "mjpc/utils/mjpc_ctrl_util.h"
#include "mjpc/utils/mjpc_math_util.h"
#include "mjpc/utilities.h"
#include "mjpc/planners/lsqp/lsqp_planner.h"
#include "mjpc/tasks/lsqp/lsqp.h"

// mjapp
#include "mjapp/robot_model.h"

namespace mjapp {
namespace mju = ::mujoco::sample_util;

// Simulate states not contained in MuJoCo structures
class Simulate : public mjpc::SimulateBase {
public:
  Simulate(std::unique_ptr<mujoco::PlatformUIAdapter> platform_ui_adapter,
           mjvCamera* cam, mjvOption* opt, mjvPerturb* pert, bool is_passive) :
    mjpc::SimulateBase(std::move(platform_ui_adapter), cam, opt, pert, nullptr, is_passive)
#if MJPC_LSQP_MANUAL_MODE
    , robot_models_({{mjpc::MAIN_ROBOT_MODEL_NAME, std::make_shared<RobotModel>(
                          mjpc::MAIN_ROBOT_MODEL_NAME, mjpc::MAIN_ROBOT_MODEL_PATH,
                          mjpc::MAIN_ROBOT_BASE_LINK_NAME,
                          mjpc::MAIN_ROBOT_EE_NAMES)}})
#endif
  {
  }

  std::map<std::string, RobotModelPtr> GetRobotModels() const { return robot_models_; }

  mjModel* ConstructCustomModel() {
    // Construct programmingly the model
    mjModel* model = lsqp_task_->ConstructModel();
    if (model) {
      // Configure model (timestep, gravity, etc.)
      lsqp_task_->ConfigureModel(model);

      // Initialize model with planner-specific infra, etc.
      lsqp_task_->Initialize(model);
    }
    return model;
  }

  void InitInThread(mjModel* model, mjData* data) override {
    // Home pos
    mj_resetDataKeyframe(model, data, mjpc::QueryKeyId(model, "home"));

    // Configure visualization
    // Site groups
    for (auto i = 0; i < model->nsite; ++i) {
      opt.sitegroup[mjMAX(0, mjMIN(mjNGROUP-1, model->site_group[i]))] = true;
    }
  }

  void PosLoadInit(const mjModel* model, const mjData* data) {
    if (mjpc::IsIIWA14Allegro()) {
      for (const auto& fingertip_name : mjpc::ALLEGRO_EE_NAMES) {
        const auto finger_target_name = fingertip_name + "_target";
        mjpc::MoveBodyMocapToSite(model, data, finger_target_name.c_str(), fingertip_name.c_str());
      }
    }
  }

  void InitControl(const mjModel* model, const mjData* data) {
    const mjpc::MutexLock lock(mtx);
    control_inited_ = true;
    // NOTE: This must be invoked everytime a scene XML is newly or reloaded,
    // and necessarily after a call to mj_forward() which fills [data]
#if MJPC_LSQP_MANUAL_MODE
    InitControlManualMode(model, mjpc::IsIIWA14()
                                   ? mjpc::IIWA14_ACTUATED_JOINT_NAMES
                                   : mjpc::IsUR5()
                                   ? mjpc::UR5_ACTUATED_JOINT_NAMES
                                   : mjpc::IsPanda()
                                   ? mjpc::PANDA_ACTUATED_JOINT_NAMES
                                   : std::vector<std::string>{});
#else
    // 1- Init [lsqp_task_]
    lsqp_task_->model_ = model;
    lsqp_task_->data_ = data;
    lsqp_task_->SetPlanner(lsqp_planner_.get());

    // 2- Init [lsqp_planner_] with [lsqp_task_]
    lsqp_planner_->Initialize(model, lsqp_task_);
#endif
  }

  void InitControlManualMode(const mjModel* model, const std::vector<std::string>& actuated_joint_names) {
    // [jnt_ids_, dof_ids, actuator_ids_]
    jnt_ids_.clear();
    dof_ids_.clear();
    actuator_ids_.clear();
    for (const auto& jnt_name : actuated_joint_names) {
      // NOTE: Here we assume all joints = dofs = actuators & sharing the same names as configured in XML
      const char* name = jnt_name.c_str();
      jnt_ids_.push_back(mjpc::QueryJointId(model, name));
      dof_ids_.push_back(mjpc::QueryDofId(model, name));
      actuator_ids_.push_back(mjpc::QueryActuatorId(model, name));
    }
  }

  static Eigen::Vector2d Circle(double t, double r, double h, double k, double f) {
    // Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    // as a function of time t and frequency f.
    auto x = r * cos(2 * M_PI * f * t) + h;
    auto y = r * sin(2 * M_PI * f * t) + k;
    return {x, y};
  }

  void Control(const mjModel* model, mjData* data, bool auto_move_target = true) override {
    const mjpc::MutexLock lock(mtx);
#if MJPC_LSQP_MANUAL_MODE
    // [Control()] can only run after [InitControl()]
    if (!control_inited_) {
      return;
    }

    if (auto_move_target && !mjpc::MAIN_ROBOT_NULLSPACE_CONTROL_ENABLED) {
      // Move [target]
      mjtNum target_pos[3];
      mju_copy3(target_pos, mjpc::QueryBodyMocapPos(model, data, "target"));
      mju_copy(target_pos, Circle(data->time, 0.1, 0.5, 0.0, 0.5).data(), 2);
      mjpc::SetBodyMocapPos(model, data, "target", target_pos);
    }

    // Control robot [ee] to track [target]
    std::vector<mjtNum> cur_qpos;
    for (const auto& jnt_id : jnt_ids_) {
      cur_qpos.push_back(mjpc::QuerySingleJointPos(model, data, jnt_id));
    }
    auto qvel = mjpc::ControlDiff(model, data,
                                  mjpc::MAIN_ROBOT_EE_SITE_NAMES[0], "target",
                                  cur_qpos.data(),
                                  mjpc::INTEGRATION_DT,
                                  mjpc::MAIN_ROBOT_NULLSPACE_CONTROL_ENABLED
                                    ? mjpc::QueryKeyJointPositions(model, "home").data()
                                    : nullptr);
    assert(cur_qpos.size() == qvel.size());
    mju_addToScl(cur_qpos.data(), qvel.data(), mjpc::INTEGRATION_DT, cur_qpos.size());
    for (auto i = 0; i < jnt_ids_.size(); ++i) {
      const auto ctrl_id = actuator_ids_[i];
      const auto ctrl_val = cur_qpos[jnt_ids_[i]];
      data->ctrl[ctrl_id] = ctrl_val;
      // ghost robot control (required for only certain experiments)
      //main_robot_model->ApplyCtrl(ctrl_id, ctrl_val);
    }
#else
    const auto eetarget_mocap_name = lsqp_task_->EETargetMocapName();
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

    if (auto_move_target) {
#if 1
      mjtNum target_pos[3];
      // MOVE [ee_target] MOCAP TO A SPECIFIC EE_TARGET-POSE
      mju_add3(target_pos, mjpc::QueryBodyPos(model, data, mjpc::Lsqp::TARGET_OBJ_NAME),
               (double [3]){0, 0, 0.07});
#else
      // MOVE [ee_target] MOCAP AROUND
      static constexpr float radius = 0.5;
      mjtNum target_pos[3] = {
          lsqp_task_->initial_ee_target_pos[0] /* + radius * cos(M_PI * data->time)*/,
          lsqp_task_->initial_ee_target_pos[1] + radius * sin(M_PI * data->time),
          lsqp_task_->initial_ee_target_pos[2]};
#endif
      mjpc::SetBodyMocapPos(model, data, eetarget_mocap_name.data(), target_pos);
      mjpc::SetBodyMocapQuat(model, data, eetarget_mocap_name.data(), mjpc::Lsqp::EE_TARGET_PREGRASP_QUAT);
    }

    // Follow [ee_target] by diff-ik
    const bool interactive = !auto_move_target;
    lsqp_planner_->LsqpControl(interactive
                                 ? nullptr
                                 : std::vector<double>(mjpc::CEM_PARAMS_TOTAL_DIM,
                                                       mjpc::Random::rand(-1., 1.)).data());
#endif
  }

  void ControlOSCBodyChain(const mjModel* model, mjData* data,
                           const std::string& ee_site_name, const std::string& target_name,
                           const std::vector<string>& jnt_names,
                           const mjtNum* key_qpos,
                           const VectorXd& kp_null, // Impedance control gains
                           const std::string& base_body_name = {}/*Make sure base_body has at least 1 dof*/) {
    std::vector<mjtNum> cur_qpos;
    std::vector<mjtNum> cur_qvel;
    std::vector<mjtNum> cur_qfrc_bias;
    // Re-init [jnt_ids_, dof_ids_, actuator_ids_]
    if (jnt_ids_[0] != mjpc::QueryJointId(model, jnt_names[0].data())) {
      InitControlManualMode(model, jnt_names);
    }
    for (const auto& jnt_id : jnt_ids_) {
      cur_qpos.push_back(mjpc::QuerySingleJointPos(model, data, jnt_id));
    }
    for (const auto& dof_id : dof_ids_) {
      cur_qvel.push_back(mjpc::QuerySingleJointVel(model, data, dof_id));
      cur_qfrc_bias.push_back(data->qfrc_bias[dof_id]);
    }
    assert(jnt_ids_.size() == dof_ids_.size());

    // Control arm -> EE
    const auto tau = mjpc::ControlOSC(model, data,
                                      ee_site_name, target_name,
                                      key_qpos,
                                      cur_qpos.data(),
                                      cur_qvel.data(),
                                      cur_qfrc_bias.data(),
                                      kp_null,
                                      mjpc::INTEGRATION_DT,
                                      base_body_name,
                                      true);
    for (auto i = 0; i < dof_ids_.size(); ++i) {
      const auto& ctrl_id = actuator_ids_[i];
      const auto& ctrl_val = tau[i];
      data->ctrl[ctrl_id] = ctrl_val;
      // ghost robot control (required for only certain experiments)
      //main_robot_model->ApplyCtrl(ctrl_id, ctrl_val);
    }
  }

  void ControlOSC(const mjModel* model, mjData* data, bool auto_move_target = false) {
    const mjpc::MutexLock lock(mtx);
#if MJPC_LSQP_MANUAL_MODE
    // [Control()] can only run after [InitControl()]
    if (!control_inited_) {
      return;
    }

    if (auto_move_target) {
      // Move [target]
      mjtNum target_pos[3];
      mju_copy3(target_pos, mjpc::QueryBodyMocapPos(model, data, "target"));
      mju_copy(target_pos, Circle(data->time, 0.1, 0.5, 0.0, 0.5).data(), 2);
      mjpc::SetBodyMocapPos(model, data, "target", target_pos);
    }

    // Control [EE] to track [target]
    // Impedence-ctrl gains
    static const mjpc::Vector7d IIWA14_KP_NULL = {75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0};
    ControlOSCBodyChain(model, data,
                        mjpc::IIWA14_EE_SITE_NAMES[0],
                        "target",
                        mjpc::IIWA14_ACTUATED_JOINT_NAMES,
                        mjpc::IIWA14_HOME_QPOS,
                        IIWA14_KP_NULL,
                        mjpc::IIWA14_BASE_DOF_BODY_NAME);

    // Control fingers -> Fingertips
    if (mjpc::IsIIWA14Allegro()) {
      static const VectorXd ALLEGRO_KP_NULL = VectorXd::Constant(mjpc::ALLEGRO_SINGLE_FINGER_QPOS_SIZE, 1.);
      for (uint8_t i = 0; i < mjpc::ALLEGRO_HOME_QPOS.size(); ++i) {
        ControlOSCBodyChain(model, data,
                            mjpc::ALLEGRO_EE_SITE_NAMES[i],
                            mjpc::ALLEGRO_EE_SITE_NAMES[i] + "_target",
                            mjpc::ALLEGRO_ACTUATED_JOINT_NAMES[i],
                            mjpc::ALLEGRO_HOME_QPOS[i],
                            ALLEGRO_KP_NULL,
                            mjpc::ALLEGRO_BASE_DOF_BODY_NAME);
      }
    }
#endif
  }

protected:
  void ModifyVisualScene(mjvScene* scn, const mjModel* model, const mjData* data) override {
#if !MJPC_VISUAL_DEBUG
    return;
#endif
    lsqp_planner_->Traces(scn);
  }

private:
  bool control_inited_ = false;
  std::map<std::string, RobotModelPtr> robot_models_;
#if MJPC_LSQP_MANUAL_MODE
  std::vector<int> jnt_ids_; // nq
  std::vector<int> dof_ids_; // nv
  std::vector<int> actuator_ids_; // nu
#endif
  mjpc::LsqpPtr lsqp_task_ = std::make_shared<mjpc::Lsqp>();
  mjpc::LsqpPlannerPtr lsqp_planner_ = std::make_shared<mjpc::LsqpPlanner>(mjpc::MjOwnerAppType::MJAPP);
};
} // namespace mujoco
