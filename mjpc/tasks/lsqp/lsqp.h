#pragma once
#include <memory>
#include <string>

// mujoco
#include <mujoco/mujoco.h>

// mjpc
#include "mjpc/task.h"
#include "mjpc/tasks/mpl/mpl_grasp_cost.h"
#include "mjpc/utilities.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/lsqp/lsqp_common.h"

#define MJPC_LSQP_PLANAR_ROBOT (0)

static const std::string MUJOCO_DIR =
#if 1
    "/home/tad/1_MUJOCO";
#else
    "/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f891/MUJOCO";
#endif
static const std::string MAIN_ROBOT_MODEL_PATH = MUJOCO_DIR +
#if MJPC_PLANNER_LSQP_DIFFIK_ENABLED
                                                 //"/mjctrl/franka_emika_panda/panda_nohand.xml";
                                                 "/mjctrl/universal_robots_ur5e/ur5e.xml";
#else
                                                 "/mink/examples/arm_hand_iiwa_allegro.xml";
#endif
static const std::string MAIN_ROBOT_MODEL_NAME = std::filesystem::path(MAIN_ROBOT_MODEL_PATH).stem();

static bool RobotHasHand() {
  return !absl::StrContainsIgnoreCase(MAIN_ROBOT_MODEL_PATH, "nohand");
}

static bool IsUR5() {
  return absl::StrContainsIgnoreCase(MAIN_ROBOT_MODEL_PATH, "ur5");
}

static const std::vector<mjtNum> IIWA14_ALLEGRO_HOME_QPOS = {
    // iiwa.
    -0.0759329, 0.153982, 0.104381, -1.8971, 0.245996, 0.34972, -0.239115,
    // allegro.
    -0.0694123, 0.0551428, 0.986832, 0.671424,
    -0.186261, -0.0866821, 1.01374, 0.728192,
    -0.218949, -0.0318307, 1.25156, 0.840648,
    1.0593, 0.638801, 0.391599, 0.57284
};
static const std::vector<mjtNum> TARGET_OBJ_HOME_QPOS = {0.5, 0, 0.5, 1, 0, 0, 0};

static constexpr uint8_t IIWA14_DOF = 7;
static constexpr uint8_t ALLEGRO_DOF = 16;

static const std::string MAIN_SCENE_XML_PATH =
#if MJPC_PLANNER_LSQP_DIFFIK_ENABLED
    MUJOCO_DIR + std::string(IsUR5()
                               ? "/mjctrl/universal_robots_ur5e/scene.xml"
                               : "/mjctrl/franka_emika_panda/scene.xml");
#else
    {};
#endif

static const std::string MAIN_ROBOT_BASE_LINK_NAME = "link0";
static const std::vector<std::string> MAIN_ROBOT_EE_NAMES = {
    IsUR5() ? "wrist_3_link" : RobotHasHand() ? "hand" : "attachment"};

static const bool MAIN_ROBOT_HAS_NULLSPACE = absl::StrContainsIgnoreCase(MAIN_ROBOT_MODEL_NAME, "panda");
// NOTE: Not necessarily the same as [model->opt.timestep]
static const double INTEGRATION_DT =
#if MJPC_PLANNER_LSQP_DIFFIK_ENABLED
    MAIN_ROBOT_HAS_NULLSPACE ? 0.1 : 1;
#else
    0.01;
#endif

namespace mjpc {
class LsqpPlanner;

class Lsqp : public Task {
public:
  Lsqp() : residual_(this) {
    mj_defaultVFS(vfs_.get());
    for (const auto& [_, spec] : spec_map_) {
      mj_deleteSpec(spec);
    }
  }

  std::string Name() const override { return "Lsqp"; }

  std::string XmlPath() const override {
    // This can be empty if [ComposeOverrideModel()] is defined
    return {};
  }

  void SetPlanner(Planner* planner) override;

  bool IsLSQPSupported() const override { return true; }
  std::string AttachmentPrefix() const { return attach_prefix_; }
  std::string AttachmentSuffix() const { return attach_suffix_; }
  std::vector<std::string> FingertipNames() const { return fingertip_names_; }
  std::map<std::string, std::vector<float>> FingertipRGBAs() const { return fingertip_rgbas_; }

  mjModel* ConstructModel() override {
#if MJPC_LSQP_PLANAR_ROBOT
    char error[1024];
    mjModel* model = mj_loadXML((MUJOCO_DIR + "/mink/examples/planar_robot/scene.xml").c_str(), vfs_.get(),
                                error, 1024);
#else
    // NOTE: For [MjcfModel::LoadToSpec()] to work, these must be independent xml files
    // -> Having no nested xml files included in themselves
    // If ones need to load from a compound xml, please use [MjcfModel::FromMjcfFile()]
    const std::string SCENE_XML =
        MUJOCO_DIR + "/mink/examples/kuka_iiwa_14/scene.xml";
    const std::string scene_name = std::filesystem::path(SCENE_XML).stem();
    const std::string KUKA_IIWA_14_XML =
        MUJOCO_DIR + "/mink/examples/kuka_iiwa_14/iiwa14.xml";
    const std::string iiwa14_name = std::filesystem::path(KUKA_IIWA_14_XML).stem();
    const std::string ALLEGRO_HAND_XML =
        MUJOCO_DIR + "/mink/examples/wonik_allegro/left_hand.xml";
    const std::string allegro_name = std::filesystem::path(ALLEGRO_HAND_XML).stem();

    // 1- Create [scene_spec]
    auto* scene_spec = CreateSpecFromXML(scene_name, SCENE_XML);
    if (!scene_spec) { return nullptr; }
    //scene_spec->memory = 10000000000;
    if (auto* key_home = mjs_findElement(scene_spec, mjOBJ_KEY, "home")) {
      mjs_delete(key_home);
    }

#if 0
    for (auto i = 1; i <= IIWA14_DOF; i++) {
      auto* act_spec = mjs_findElement(scene_spec, mjOBJ_ACTUATOR, ("actuator" + std::to_string(i)).c_str());
      if (auto* act = mjs_asActuator(act_spec)) {
        mju_copy3(act->biasprm, (double[]){0, -2000, -200});
      }
    }
#endif

    // 2- Create [allegro_spec]
    auto* allegro_spec = CreateSpecFromXML(allegro_name, ALLEGRO_HAND_XML);
    if (!allegro_spec) { return nullptr; }
    const auto allegro_model_name = std::string(mjs_getString(allegro_spec->modelname));
    mjsBody* allegro_palm = mjs_findBody(allegro_spec, "palm");
    mju_copy4(allegro_palm->quat, mjpc::ROTATION_IDENTITY);
    memcpy(allegro_palm->pos, (mjtNum[]){0.0, 0.0, 0.095}, sizeof(allegro_palm->pos));

    // 3- Attach [allegro_palm] -> [scene_spec] through [attach_site]
    // NOTE: This prefix will be prepended to names of all child elements (bodies, geoms, etc.) in [allegro_model]
    attach_prefix_ = allegro_model_name + "/";
    mjsSite* attach_site = mjpc::FindSiteSpec(scene_spec, "attachment_site");
    mjs_attachToSite(attach_site, allegro_palm, attach_prefix_.c_str(), attach_suffix_.c_str());

    // Re-key new assembled robot model [scene_spec]
    mjsKey* new_key = mjs_addKey(scene_spec);
    mjs_setString(new_key->name, "home");
    // [iiwa14] - vel actuator: init qpos
    std::vector<mjtNum> total_qpos = IIWA14_ALLEGRO_HOME_QPOS;
    total_qpos.insert(total_qpos.end(), TARGET_OBJ_HOME_QPOS.begin(), TARGET_OBJ_HOME_QPOS.end());
    mjs_setDouble(new_key->qpos, total_qpos.data(), total_qpos.size());
    // [allegro] - pos actuator: init ctrl
    mjs_setDouble(new_key->ctrl, IIWA14_ALLEGRO_HOME_QPOS.data(),
                  IIWA14_ALLEGRO_HOME_QPOS.size());
    mju_zero(new_key->ctrl->data(), IIWA14_ALLEGRO_HOME_QPOS.size() - ALLEGRO_DOF);

    // Add fingertip-mocap bodies
    mjsBody* world_body = mjpc::FindWorldBodySpec(scene_spec);
    for (const auto& fingertip : fingertip_names_) {
      mjsBody* finger_mocap = mjs_addBody(world_body, nullptr);
      mjs_setString(finger_mocap->name, (attach_prefix_ + fingertip + "_target").c_str());
      finger_mocap->mocap = true;
      mjsGeom* geom = mjs_addGeom(finger_mocap, nullptr);
      geom->type = mjGEOM_SPHERE;
      memcpy(geom->size, (mjtNum[]){0.02, 0.02, 0.02}, sizeof(geom->size));
      memcpy(geom->rgba, fingertip_rgbas_[fingertip].data(), sizeof(geom->rgba));
      // Disable collision
      geom->contype = 0;
      geom->conaffinity = 0;
    }

    // Add user sensor as residuals
    mjsSensor* user_sensor = mjs_addSensor(scene_spec);
    mjs_setString(user_sensor->name, "Grasp");
    user_sensor->dim = 4;
    user_sensor->type = mjSENS_USER;
    mjs_setDouble(user_sensor->userdata, (double[]){1, 75, 0, 100, 0.02, 2}, 6);

    //!NOTE: If needed, consider setting [scene_spec->modelfiledir], base for all resource paths
    mjModel* model = mj_compile(scene_spec, vfs_.get());
#if MJPC_PLANNER_LSQP_DEBUG
    if (model) {
      // Output to bin folder
      mj_saveXML(scene_spec, (iiwa14_name + "_" + allegro_name + ".xml").c_str(), nullptr, 0);
    }
#endif
#endif
    return model;
  }

  void ConfigureModel(mjModel* model) override {
    // Configure [model]
    //model->opt.disableactuator |= (1 << 0);
    model->opt.timestep = 0.002;
    // Disable gravity
    static constexpr float kGravity = 0.; // -9.81f;
    model->opt.gravity[2] = kGravity;
    if (kGravity > 0.0) {
      // Enable gravity compensation
      for (auto i = 0; i < model->nbody; ++i) {
        model->body_gravcomp[i] = true;
      }
    }
  }

  void InitMocaps() const {
#if MJPC_LSQP_PLANAR_ROBOT
    MoveBodyMocapToSite("target_mocap", "hand");
#else
    MoveBodyMocapToSite("target", "attachment_site");
    for (const auto& fingertip_name : FingertipNames()) {
      const auto fingertip_target = attach_prefix_ + fingertip_name + "_target";
      assert(QueryBodyMocapId(fingertip_target.c_str()) >= 0);
      const auto finger_site_name = attach_prefix_ + fingertip_name;
      if (const auto finger_site_id = QuerySiteId(finger_site_name.c_str())) {
        MoveBodyMocapToSite(fingertip_target.c_str(), finger_site_id);
      }
    }
#endif
  }

  class ResidualFn : public BaseResidualFn {
  public:
    explicit ResidualFn(const Lsqp* task) : BaseResidualFn(task) {
    }

    void Residual(const mjModel* model, const mjData* data, double* residual) const override;
  };

  void ResetLocked(const mjModel* model) override {
    Task::ResetLocked(model);
    mocaps_inited_ = false;
  }

  // Reset the cube into the hand if it's on the floor
  void TransitionLocked(mjModel* model, mjData* data) override;

protected:
  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }

  ResidualFn* InternalResidual() override {
    return &residual_;
  }

  //! xml: xml string or file path
  mjSpec* CreateSpecFromXML(const std::string& spec_name, const std::string& xml) {
    if (auto* spec = mjpc::MjcfModel::LoadToSpec(xml, vfs_.get())) {
      if (spec_map_.contains(spec_name)) {
        mj_deleteSpec(spec_map_[spec_name]);
      }
      spec_map_.emplace(spec_name, spec);
      return spec;
    }
    return nullptr;
  }

private:
  LsqpPlanner* lsqp_planner_ = nullptr;
  ResidualFn residual_;
  std::vector<std::string> fingertip_names_ = {"rf_tip", "mf_tip", "ff_tip", "th_tip"};
  std::map<std::string, std::vector<float>> fingertip_rgbas_ = {
      {fingertip_names_[0], {0.9f, 0.f, 0.f, 1.f}}, // Red
      {fingertip_names_[1], {0.f, 0.9f, 0.f, 1.f}}, // Green
      {fingertip_names_[2], {0.f, 0.f, 0.9f, 1.f}}, // Blue
      {fingertip_names_[3], {0.9f, 0.9f, 0.9f, 1.f}}}; // White
  bool mocaps_inited_ = false;
  std::unique_ptr<mjVFS> vfs_ = std::make_unique<mjVFS>();
  std::map<std::string, mjSpec*> spec_map_;
  std::string attach_prefix_;
  std::string attach_suffix_;
};
} // namespace mjpc
