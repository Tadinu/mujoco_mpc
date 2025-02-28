#pragma once

#include <atomic>
#include <chrono>
#include <utility>
#include <filesystem>

// Eigen
#include <Eigen/Core>

// MuJoCo
#include <mujoco/mujoco.h>

#include <fmt/format.h>

// mjpc
#include "mjpc/utils/mjpc_ctrl_util.h"
#include "mjpc/utilities.h"
#include "mjpc/planners/lsqp/lsqp_planner.h"
#include "mjpc/planners/fabrics/include/fab_math_util.h"
#include "mjpc/tasks/lsqp/lsqp.h"

// mjapp
#include "mjapp/sim_base.h"
#include "mjapp/robot_model.h"
#include "mjapp/array_safety.h"

namespace mjapp {
namespace mjapp_util = ::mjapp::sample_util;

// Simulate states not contained in MuJoCo structures
class Simulate : public SimulateBase {
public:
  Simulate(std::unique_ptr<PlatformUIAdapter> platform_ui_adapter,
           mjvCamera* cam, mjvOption* opt, mjvPerturb* pert, bool is_passive) :
    SimulateBase(std::move(platform_ui_adapter), cam, opt, pert, is_passive)
#if MJPC_PLANNER_LSQP_DIFFIK_ENABLED
    ,robot_models_({{MAIN_ROBOT_MODEL_NAME, std::make_shared<RobotModel>(
                        MAIN_ROBOT_MODEL_NAME, MAIN_ROBOT_MODEL_PATH, MAIN_ROBOT_BASE_LINK_NAME,
                        MAIN_ROBOT_EE_NAMES)}}),
#endif
  {
  }

  std::map<std::string, RobotModelPtr> GetRobotModels() const { return robot_models_; }

  mjModel* ConstructCustomModel() {
    // Construct programmingly the model
    mjModel* model = lsqp_task_.ConstructModel();
    if (model) {
      // Configure model (timestep, gravity, etc.)
      lsqp_task_.ConfigureModel(model);

      // Initialize model with planner-specific infra, etc.
      lsqp_task_.Initialize(model);
    }
    return model;
  }

  void Init(mjModel* model, mjData* data) override {
    // Home pos
    mj_resetDataKeyframe(model, data, mjpc::QueryKeyId(model, "home"));

    // Configure visualization
    // Site groups
    for (auto i = 0; i < model->nsite; ++i) {
      opt.sitegroup[mjMAX(0, mjMIN(mjNGROUP-1, model->site_group[i]))] = true;
    }
  }

  void InitControl(mjModel* model, mjData* data) {
    // NOTE: This must be invoked everytime a scene XML is newly or reloaded,
    // and necessarily after a call to mj_forward() which fills [data]
#if MJPC_PLANNER_LSQP_DIFFIK_ENABLED
    const auto& main_robot_model = robot_models_[MAIN_ROBOT_MODEL_NAME];
    // [jnt_ids_, actuator_ids_]
    jnt_ids_.clear();
    actuator_ids_.clear();
    for (const auto& jnt_name : main_robot_model->JointNames()) {
      // NOTE: Here we assume all joints = dofs = actuators & sharing the same names as configured in MXML
      const char* name = jnt_name.c_str();
      jnt_ids_.push_back(mjpc::QueryJointId(model, name));
      actuator_ids_.push_back(mjpc::QueryActuatorId(model, name));
    }
#else
    // 1- Init [lsqp_task_]
    lsqp_task_.model_ = model;
    lsqp_task_.data_ = data;
    lsqp_task_.SetPlanner(lsqp_planner_.get());

    // 2- Init [lsqp_planner_] with [lsqp_task_]
    lsqp_planner_->Initialize(model, lsqp_task_);
#endif
  }

  static Eigen::Vector2d Circle(double t, double r, double h, double k, double f) {
    // Return the (x, y) coordinates of a circle with radius r centered at (h, k)
    // as a function of time t and frequency f.
    auto x = r * cos(2 * M_PI * f * t) + h;
    auto y = r * sin(2 * M_PI * f * t) + k;
    return {x, y};
  }

  void Control(const mjModel* model, mjData* data) override {
    //const std::unique_lock<std::recursive_mutex> lock(mtx);
#if MJPC_PLANNER_LSQP_DIFFIK_ENABLED
    const auto& main_robot_model = robot_models_[MAIN_ROBOT_MODEL_NAME];
    const int nv = model->nv;
    assert(nv == main_robot_model->JointNames().size());
    if (!MAIN_ROBOT_HAS_NULLSPACE) {
      // Move [target]
      mjtNum target_pos[3];
      mju_copy3(target_pos, mjpc::QueryBodyMocapPos(model, data, "target"));
      mju_copy(target_pos, Circle(data->time, 0.1, 0.5, 0.0, 0.5).data(), 2);
      mjpc::SetBodyMocapPos(model, data, "target", target_pos);
    }

    // Control robot [ee] to track [target]
    std::vector<mjtNum> cur_qpos;
    for (auto i = 0; i < jnt_ids_.size(); ++i) {
      cur_qpos.push_back(mjpc::QuerySingleJointPos(model, data, jnt_ids_[i]));
    }
    auto qvel = mjpc::ControlDiff(model, data, MAIN_ROBOT_BASE_LINK_NAME.c_str(),
                                  MAIN_ROBOT_EE_NAMES[0].c_str(), "target",
                                  cur_qpos.data(),
                                  INTEGRATION_DT,
                                  MAIN_ROBOT_HAS_NULLSPACE);

    auto qpos = std::vector<mjtNum>(data->qpos, data->qpos + nv);
    mju_addToScl(qpos.data(), qvel.data(), INTEGRATION_DT, nv);
    for (auto i = 0; i < jnt_ids_.size(); ++i) {
      const auto ctrl_id = actuator_ids_[i];
      const auto ctrl_val = qpos[jnt_ids_[i]];
      data->ctrl[ctrl_id] = ctrl_val;
      main_robot_model->ApplyCtrl(ctrl_id, ctrl_val);
    }
#else
    // Move [ee_target] around
    static constexpr float radius = 0.5;
#if 0
    double pos[3] = {
      lsqp_task_.initial_ee_target_pos[0] /* + radius * cos(M_PI * data->time)*/,
      lsqp_task_.initial_ee_target_pos[1] + radius * sin(M_PI * data->time),
      lsqp_task_.initial_ee_target_pos[2]};
#else
    double pos[3];
#endif

    const auto eetarget_mocap_name = lsqp_task_.EETargetMocapName();
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

#if 1
    // MOVE TO A SPECIFIC EE_TARGET-POSE
    mjtNum target_pos[3];
    mju_add3(target_pos, mjpc::QueryBodyPos(model, data, mjpc::Lsqp::TARGET_OBJ_NAME),
             (double [3]){0, 0, 0.07});
    mjpc::SetBodyMocapPos(model, data, eetarget_mocap_name.data(), target_pos);
    mjpc::SetBodyMocapQuat(model, data, eetarget_mocap_name.data(), mjpc::Lsqp::EE_TARGET_PREGRASP_QUAT);
#endif
    // Follow [ee_target] by diff-ik
    constexpr bool interactive = true;
    lsqp_planner_->LsqpControl(interactive
                                 ? nullptr
                                 : std::vector<double>(mjpc::CEM_PARAMS_TOTAL_DIM,
                                                       FabRandom::rand(-1., 1.)).data());
#endif
  }

protected:
  void ModifyVisualScene(mjvScene* scn, const mjModel* model, const mjData* data) override {
#if !MJAPP_VISUAL_DEBUG
    return;
#endif
    lsqp_planner_->Traces(scn);
  }

private:
  std::map<std::string, RobotModelPtr> robot_models_;
#if MJPC_PLANNER_LSQP_DIFFIK_ENABLED
  std::vector<int> jnt_ids_;
  std::vector<int> actuator_ids_;
#endif
  mjpc::Lsqp lsqp_task_;
  mjpc::LsqpPlannerPtr lsqp_planner_ = std::make_shared<mjpc::LsqpPlanner>(mjpc::MjOwnerAppType::MJAPP);
};
} // namespace mujoco
