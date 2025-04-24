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
#include "mjpc/tasks/lsqp/iiwa14_allegro.h"
#include "mjpc/tasks/garmi/garmi.h"

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
    InitControlManualMode(model, mjpc::MAIN_ACTUATED_JOINT_NAMES, mjpc::MAIN_ACTUATOR_NAMES);
#else
    // 1- Init [lsqp_task_]
    lsqp_task_->model_ = const_cast<mjModel*>(model);
    lsqp_task_->data_ = const_cast<mjData*>(data);
    lsqp_task_->SetPlanner(lsqp_planner_.get());

    // 2- Init [lsqp_planner_] with [lsqp_task_]
    lsqp_planner_->Initialize(lsqp_task_->model_, *lsqp_task_);
#endif
  }

#if MJPC_LSQP_MANUAL_MODE
  void InitControlManualMode(const mjModel* model, const std::vector<std::string>& actuated_joint_names,
                             std::vector<std::string> actuator_names = {}) {
    // [jnt_ids_, dof_ids, actuator_ids_]
    jnt_ids_.clear();
    dof_ids_.clear();
    actuator_ids_.clear();
    for (const auto& jnt_name : actuated_joint_names) {
      // NOTE: Here we assume all joints = dofs & sharing the same names as configured in XML
      const char* name = jnt_name.c_str();
      jnt_ids_.push_back(mjpc::QueryJointId(model, name));
      dof_ids_.push_back(mjpc::QueryDofId(model, name));
    }

    if (actuator_names.empty()) {
      actuator_names = actuated_joint_names;
    }

    for (const auto& act_name : actuator_names) {
      actuator_ids_.push_back(mjpc::QueryActuatorId(model, act_name.c_str()));
    }
  }

  static Eigen::Vector2d Circle(double t, double r, double h, double k, double f) {
    // Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    // as a function of time t and frequency f.
    auto x = r * cos(2 * M_PI * f * t) + h;
    auto y = r * sin(2 * M_PI * f * t) + k;
    return {x, y};
  }
#endif

  void Control(const mjModel* model, mjData* data, bool kinematics_only = MJPC_LSQP_KINEMATICS_ONLY,
               bool auto_move_target = false) override {
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
#if 0
      mjtNum target_quat[4];
      for (int i = 0; i < 4; i++) {
        target_quat[i] = mjpc::Random::rand();
      }
      mju_normalize4(target_quat);
      mjpc::SetBodyMocapQuat(model, data, "target", target_quat);
#endif
    }

    // Control robot [ee] to track [target]
    if (last_qpos_.empty()) {
      last_qpos_ = mjpc::QueryKeyJointPositions(model, "home");
      if (last_qpos_.empty()) {
        last_qpos_ = std::vector<mjtNum>(jnt_ids_.size(), 0.);
      }
    }
    std::vector<mjtNum> cur_qpos;
    for (const auto& jnt_id : jnt_ids_) {
      cur_qpos.push_back(mjpc::QuerySingleJointPos(model, data, jnt_id));
    }

    VectorXd qvel;
    if (mjpc::IsDualPanda()) {
      static constexpr int NV = 14;
      static constexpr int dofs = NV / 2;
      using Vector14d = Eigen::Matrix<double, NV, 1>;
      qvel = Vector14d::Zero();
      qvel.head<7>() = mjpc::ControlDiff(model, data, mjpc::MAIN_ROBOT_EE_SITE_NAMES[0], "target",
                                         cur_qpos.data(),
                                         mjpc::INTEGRATION_DT,
                                         mjpc::MAIN_ROBOT_NULLSPACE_CONTROL_ENABLED
                                           ? last_qpos_.data()
                                           : nullptr,
                                         mjpc::IsBiFrankaPanda() ? "panda0_link0" : "left_link0");
      qvel.tail<7>() = mjpc::ControlDiff(model, data, mjpc::MAIN_ROBOT_EE_SITE_NAMES[1], "target",
                                         cur_qpos.data() + dofs,
                                         mjpc::INTEGRATION_DT,
                                         mjpc::MAIN_ROBOT_NULLSPACE_CONTROL_ENABLED
                                           ? (last_qpos_.data() + dofs)
                                           : nullptr,
                                         mjpc::IsBiFrankaPanda() ? "panda1_link0" : "right_link0");
    } else {
      qvel = mjpc::ControlDiff(model, data,
                               mjpc::MAIN_ROBOT_EE_SITE_NAMES[0], "target",
                               cur_qpos.data(),
                               mjpc::INTEGRATION_DT,
                               mjpc::MAIN_ROBOT_NULLSPACE_CONTROL_ENABLED ? last_qpos_.data() : nullptr);
    }
    assert(cur_qpos.size() == qvel.size());
    mju_addToScl(cur_qpos.data(), qvel.data(), mjpc::INTEGRATION_DT, cur_qpos.size());
    for (auto i = 0; i < jnt_ids_.size(); ++i) {
      const auto ctrl_id = actuator_ids_[i];
      const auto ctrl_val = cur_qpos[jnt_ids_[i]];
      data->ctrl[ctrl_id] = ctrl_val;
      // ghost robot control (required for only certain experiments)
      //main_robot_model->ApplyCtrl(ctrl_id, ctrl_val);
    }
    last_qpos_.assign(cur_qpos.begin(), cur_qpos.end());
#else
    // Follow [ee_target]
    const bool interactive = !auto_move_target;
    const auto ctrl = lsqp_planner_->LsqpControl(interactive
                                                   ? nullptr
                                                   : std::vector<double>(mjpc::CEM_PARAMS_TOTAL_DIM,
                                                     mjpc::Random::rand(-1., 1.)).data());
    // Copy [ctrl] -> [data->ctrl]
    if (!mjpc::AreInvalidControls(ctrl)) {
#if 1
      static const bool is_garmi = std::dynamic_pointer_cast<mjpc::Garmi>(lsqp_task_) != nullptr;
      static std::vector<std::string> finger_jnt_names;
      if (finger_jnt_names.empty() && !is_garmi) {
        for (const auto& fjnames : mjpc::ALLEGRO_ACTUATED_JOINT_NAMES) {
          finger_jnt_names = mjpc::ChainCollections<std::string>(finger_jnt_names, fjnames);
        }
      }
      static const auto joint_names = is_garmi
                                        ? mjpc::GARMI_ACTUATED_JOINT_NAMES
                                        : mjpc::ChainCollections<std::string>(
                                            mjpc::IIWA14_ACTUATED_JOINT_NAMES,
                                            finger_jnt_names);
      static const auto act_names = is_garmi
                                      ? mjpc::GARMI_ACTUATOR_NAMES
                                      : joint_names;

      if (kinematics_only) {
        for (const auto& jnt_name : joint_names) {
          const int i = mjpc::QueryJointId(model, jnt_name.c_str());
          if (mjpc::Lsqp::POSITION_CTRL_ENABLED) {
            data->qpos[mjpc::QueryJointPosAddress(model, i)] = ctrl[i];
          } else {
            data->qvel[mjpc::QueryJointDofAddress(model, i)] = ctrl[i];
          }
        }
      } else {
        for (int i = 0; i < act_names.size(); ++i) {
          const auto& act_name = act_names[i];
          data->ctrl[mjpc::QueryActuatorId(model, act_name.c_str())] = ctrl[i];
        }
      }
#else
      mju_copy(kinematics_only ? (mjpc::Lsqp::POSITION_CTRL_ENABLED ? data->qpos : data->qvel) : data->ctrl,
               ctrl.data(), ctrl.size());
#endif
    }
#endif // END MJPC_LSQP_AUTO_MODE
  }

#if MJPC_LSQP_MANUAL_OSC_ENABLED
  void ControlOSCBodyChain(const mjModel* model, mjData* data,
                           const std::string& ee_site_name, const std::string& target_name,
                           const std::vector<string>& jnt_names,
                           const mjtNum* key_qpos,
                           const VectorXd& kp_null, // Impedance control gains
                           const std::string& base_body_name = {}/*Make sure base_body has at least 1 dof*/,
                           bool kinematics_only = MJPC_LSQP_KINEMATICS_ONLY) {
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
      if (kinematics_only) {
        data->qpos[i] = ctrl_val;
      }
      else {
        data->ctrl[ctrl_id] = ctrl_val;
      }
      // ghost robot control (required for only certain experiments)
      //main_robot_model->ApplyCtrl(ctrl_id, ctrl_val);
    }
  }

  void ControlOSC(const mjModel* model, mjData* data, bool kinematics_only = MJPC_LSQP_KINEMATICS_ONLY,
    bool auto_move_target = false) {
    const mjpc::MutexLock lock(mtx);
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
                        mjpc::IIWA14_BASE_DOF_BODY_NAME,
                        kinematics_only);

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
                            mjpc::ALLEGRO_BASE_DOF_BODY_NAME,
                            kinematics_only);
      }
    }
  }
#endif

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
  std::vector<mjtNum> last_qpos_;
#endif
  mjpc::LsqpPtr lsqp_task_ = std::make_shared<mjpc::Garmi>();
  mjpc::LsqpPlannerPtr lsqp_planner_ = std::make_shared<mjpc::LsqpPlanner>(mjpc::MjOwnerAppType::MJAPP);
};
} // namespace mujoco
