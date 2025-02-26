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
#include "mjpc/planners/lsqp/lsqp_se3.h"

#define MJPC_LSQP_PLANAR_ROBOT (0)
#define MJPC_LSQP_SPAWN_OBJECT (1)
#define MJPC_LSQP_FINGERS_OSC (1)

namespace mjpc {
static const std::string MUJOCO_DIR =
#if 0
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

static std::vector<mjtNum> TARGET_OBJ_QPOS = {0.9, 0, 0.3, 1, 0, 0, 0};

static constexpr uint8_t IIWA14_DOF = 7;
static constexpr uint8_t ALLEGRO_DOF = 16;
static constexpr uint8_t EE_CEM_PARAMS_DIM = 4; // wrist(XYZ-loc + theta-rot]
static constexpr uint8_t FINGERS_CEM_PARAMS_DIM = MJPC_LSQP_FINGERS_OSC
                                                    ? 4 // Fingertips
                                                    : ALLEGRO_DOF;
static constexpr uint8_t IIWA14_ALLEGRO_CEM_PARAMS_DIM = EE_CEM_PARAMS_DIM + FINGERS_CEM_PARAMS_DIM;

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


class LsqpPlanner;

class Lsqp : public Task {
public:
  static constexpr float KGRAVITY = -9.81f;
  static constexpr bool SYSTEM_MODEL_ACTUATORS_OSC = false;
  static constexpr bool POSITION_CTRL_ENABLED = true;

  static constexpr const char* EE_TARGET_NAME = "ee_target";
  static constexpr double EE_TARGET_PREGRASP_QUAT[4] = {0.644963, -0.112683, 0.75546, 0.0246038};
  static constexpr const char* ATTACHMENT_SITE_NAME = "attachment_site";
  static constexpr const char* PALM_NAME = "palm";
  static const std::vector<std::string> FINGERTIP_NAMES;
  static const std::map<std::string, std::vector<float>> FINGERTIPS_RGBA;
  static constexpr const char* TARGET_OBJ_NAME = "target";
  static constexpr const char* TARGET_OBJ_GOAL_NAME = "target_goal";

  // NOTE: Hardcoded trace prefix as required in task.cc
  static constexpr const char* TRACE_SENSOR_PREFIX = "trace";

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

  std::string PalmBodyName() const {
    return attach_prefix_ + PALM_NAME;
  }

  std::string PalmSiteName() const {
    return PalmBodyName();
  }

  std::string FingertipBodyName(const std::string& fingertip_name) const {
    return attach_prefix_ + fingertip_name;
  }

  std::string FingertipSiteName(const std::string& fingertip_name) const {
    return FingertipBodyName(fingertip_name);
  }

  std::string FingertipTargetBodyName(const std::string& fingertip_name) const {
    return FingertipSiteName(fingertip_name) + "_target";
  }

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
    scene_spec->nuserdata = LsqpSE3::PARAMS_DIM; // Storing latest [T_ee] for LsqpPlanner's Diff-IK Control

    // 2- Create [allegro_spec]
    auto* allegro_spec = CreateSpecFromXML(allegro_name, ALLEGRO_HAND_XML);
    if (!allegro_spec) { return nullptr; }
    const auto allegro_model_name = std::string(mjs_getString(allegro_spec->modelname));
    mjsBody* allegro_palm = mjs_findBody(allegro_spec, "palm");
    mju_copy4(allegro_palm->quat, mjpc::ROTATION_IDENTITY);
    memcpy(allegro_palm->pos, (mjtNum[]){0.0, 0.0, 0.095}, sizeof(allegro_palm->pos));

    // 3- Create agents' numeric data (horizon, timestep)
    // Refer to: CrossEntropyPlanner::Initialize() for specific numeric data names
    const auto fCreateNumeric = [&scene_spec](const char* numeric_name, double value) {
      auto* numeric = mjs_addNumeric(scene_spec);
      mjs_setString(numeric->name, numeric_name);
      numeric->size = 1;
      mjs_setDouble(numeric->data, (double[]){value}, 1);
    };
    // NOTE: Larger [agent_horizon] may require larger [kMaxTrajectoryHorizon] configured in [trajectory.h]
    fCreateNumeric("agent_horizon", 0.1);
    fCreateNumeric("agent_timestep", 0.01);
    fCreateNumeric("sampling_trajectories", 10);

    // 4- Customize actuactors
    if constexpr (SYSTEM_MODEL_ACTUATORS_OSC) {
      // Delete all the default fully-actuated actuators
      for (auto i = 1; i <= IIWA14_DOF; i++) {
        auto* act_spec = mjs_findElement(scene_spec, mjOBJ_ACTUATOR,
                                         ("actuator" + std::to_string(i)).c_str());
        if (mjs_asActuator(act_spec)) {
          mjs_delete(act_spec);
        }
      }

      static std::vector ALLEGRO_ACT_NAMES = {"ffa0", "ffa1", "ffa2", "ffa3",
                                              "mfa0", "mfa1", "mfa2", "mfa3",
                                              "rfa0", "rfa1", "rfa2", "rfa3",
                                              "tha0", "tha1", "tha2", "tha3"};
      for (const auto& act_name : ALLEGRO_ACT_NAMES) {
        auto* act_spec = mjs_findElement(allegro_spec, mjOBJ_ACTUATOR, act_name);
        if (mjs_asActuator(act_spec)) {
          mjs_delete(act_spec);
        }
      }
    } else {
      for (auto i = 1; i <= IIWA14_DOF; i++) {
        auto* act_spec = mjs_findElement(scene_spec, mjOBJ_ACTUATOR,
                                         ("actuator" + std::to_string(i)).c_str());
        if (auto* act = mjs_asActuator(act_spec)) {
          act->trntype = mjTRN_JOINT;
          act->dyntype = mjDYN_NONE;
          act->biastype = mjBIAS_AFFINE;
          act->gaintype = mjGAIN_FIXED;
          constexpr double kp = 2000;
          constexpr double kv = 200;
          if constexpr (POSITION_CTRL_ENABLED) {
            mju_copy3(act->gainprm, (double[]){kp, 0, 0});
            mju_copy3(act->biasprm, (double[]){0, -kp, -kv});
          } else {
            mju_copy3(act->gainprm, (double[]){kv, 0, 0});
            mju_copy3(act->biasprm, (double[]){0, 0, -kv});
          }
        }
      }
    }

    // 5- Attach [allegro_palm] -> [scene_spec] through [attach_site]
    // NOTE: This prefix will be prepended to names of all child elements (bodies, geoms, etc.) in [allegro_model]
    attach_prefix_ = allegro_model_name + "/";
    mjsSite* attach_site = mjpc::FindSiteSpec(scene_spec, ATTACHMENT_SITE_NAME);
    mjs_attachToSite(attach_site, allegro_palm, attach_prefix_.c_str(), attach_suffix_.c_str());

    // 6- Re-key new assembled robot model [scene_spec]
    auto* key_home = mjpc::FindKeySpec(scene_spec, "home");
    if (!key_home) {
      key_home = mjs_addKey(scene_spec);
      mjs_setString(key_home->name, "home");
    }

    // 6.1- [iiwa14] - vel actuator: init qpos
    system_qpos_home = IIWA14_ALLEGRO_HOME_QPOS;
#if MJPC_LSQP_SPAWN_OBJECT
    system_qpos_home.insert(system_qpos_home.end(), TARGET_OBJ_QPOS.begin(), TARGET_OBJ_QPOS.end());
#endif
    mjs_setDouble(key_home->qpos, system_qpos_home.data(), system_qpos_home.size());

    // 6.2- [allegro] - pos actuator: init ctrl
    if constexpr (SYSTEM_MODEL_ACTUATORS_OSC) {
    } else {
      mjs_setDouble(key_home->ctrl, IIWA14_ALLEGRO_HOME_QPOS.data(),
                    IIWA14_ALLEGRO_HOME_QPOS.size());
      mju_zero(key_home->ctrl->data(), IIWA14_ALLEGRO_HOME_QPOS.size() - ALLEGRO_DOF);
    }

    // 7- Add mocap bodies
    mjsBody* world_body = mjpc::FindWorldBodySpec(scene_spec);
    const auto fCreateSite = [](mjsBody* body, const std::string& site_name,
                                mjtGeom type = mjGEOM_SPHERE, double size = 0.001) {
      mjsSite* site = mjs_addSite(body, nullptr);
      mjs_setString(site->name, site_name.c_str());
      site->type = type;
      memcpy(site->size, (mjtNum[]){size, size, size}, sizeof(site->size));
      site->group = 4;
      return site;
    };

    // 7.1- [ee-mocap body]
    mjsBody* ee_mocap = mjs_addBody(world_body, nullptr);
    mjs_setString(ee_mocap->name, EE_TARGET_NAME);
    memcpy(ee_mocap->pos, (double[]){0.5, 0, 0.5}, sizeof(ee_mocap->pos));
    memcpy(ee_mocap->quat, (double[]){0, 1, 0, 0}, sizeof(ee_mocap->quat));
    ee_mocap->mocap = true;
    mjsGeom* ee_mocap_geom = mjs_addGeom(ee_mocap, nullptr);
    ee_mocap_geom->type = mjGEOM_BOX;
    memcpy(ee_mocap_geom->size, (double[]){0.05, 0.05, 0.05}, sizeof(ee_mocap_geom->size));
    memcpy(ee_mocap_geom->rgba, (float[]){0.6, 0.5, 0.3, 0.2}, sizeof(ee_mocap_geom->rgba));
    // Disable collision
    ee_mocap_geom->contype = 0;
    ee_mocap_geom->conaffinity = 0;

    // [ee_mocap_site]/[palm_site]
    if constexpr (SYSTEM_MODEL_ACTUATORS_OSC) {
      fCreateSite(ee_mocap, EE_TARGET_NAME);
    } else {
      fCreateSite(allegro_palm, PalmSiteName());;
    }

    // 7.2- [Fingertip-mocap bodies]
    for (const auto& fingertip : FINGERTIP_NAMES) {
      const auto fingertip_target_name = FingertipTargetBodyName(fingertip);
      mjsBody* finger_mocap = mjs_addBody(world_body, nullptr);
      mjs_setString(finger_mocap->name, fingertip_target_name.c_str());
      finger_mocap->mocap = true;
      mjsGeom* finger_mocap_geom = mjs_addGeom(finger_mocap, nullptr);
      finger_mocap_geom->type = mjGEOM_SPHERE;
      memcpy(finger_mocap_geom->size, (mjtNum[]){0.02, 0.02, 0.02}, sizeof(finger_mocap_geom->size));
      memcpy(finger_mocap_geom->rgba, FINGERTIPS_RGBA.at(fingertip).data(), sizeof(finger_mocap_geom->rgba));
      // Disable collision
      finger_mocap_geom->contype = 0;
      finger_mocap_geom->conaffinity = 0;

      // [fingermocap_site]
      if constexpr (SYSTEM_MODEL_ACTUATORS_OSC) {
        fCreateSite(finger_mocap, fingertip_target_name);
      }
    }

    // 8- [Mocap site Actuators]
    if constexpr (SYSTEM_MODEL_ACTUATORS_OSC) {
      const auto fCreateActuator = [&scene_spec](const std::string& actuator_name,
                                                 const std::string& site_name,
                                                 double gear[]) {
        mjsActuator* act = mjs_addActuator(scene_spec, 0);
        mjs_setString(act->name, actuator_name.c_str());
        act->trntype = mjTRN_SITE;
        //act->dyntype = mjtDyn::mjDYN_INTEGRATOR;
        mjs_setString(act->target, site_name.c_str());
        act->ctrllimited = true;
        mju_copy(act->ctrlrange, (double[]){-1, 1}, 2);
        mju_copy(act->gear, gear, 6);
        act->gainprm[0] = 1;
      };

      // Add actuators for [ee + fingertip] targets
      fCreateActuator("ee_act_x", EE_TARGET_NAME, (double[]){1, 0, 0, 0, 0, 0});
      fCreateActuator("ee_act_y", EE_TARGET_NAME, (double[]){0, 1, 0, 0, 0, 0});
      fCreateActuator("ee_act_z", EE_TARGET_NAME, (double[]){0, 0, 1, 0, 0, 0});
      for (const auto& fingertip : FINGERTIP_NAMES) {
        const auto fingertip_mocap_site_name = FingertipTargetBodyName(fingertip);
        const auto fingertip_name = attach_prefix_ + fingertip;
        fCreateActuator(fingertip_name + "_x", fingertip_mocap_site_name,
                        (double[]){1, 0, 0, 0, 0, 0});
        fCreateActuator(fingertip_name + "_y", fingertip_mocap_site_name,
                        (double[]){0, 1, 0, 0, 0, 0});
        fCreateActuator(fingertip_name + "_z", fingertip_mocap_site_name,
                        (double[]){0, 0, 1, 0, 0, 0});
      }
    }

    // 9- [User sensors] as residuals
    // https://github.com/google-deepmind/mujoco_mpc/blob/main/docs/OVERVIEW.md#residual-specification
    // 9.1- Reach sensor
    mjsSensor* reach_sensor = mjs_addSensor(scene_spec);
    reach_sensor->type = mjSENS_USER;
    reach_sensor->dim = 3;
    mjs_setString(reach_sensor->name, "Reach");
    mjs_setDouble(reach_sensor->userdata, (double[]){0 /*Quadratic norm*/, 2.5, 0, 5, 0.01}, 5);

    // 9.2- Bring sensor
    mjsSensor* bring_sensor = mjs_addSensor(scene_spec);
    bring_sensor->type = mjSENS_USER;
    bring_sensor->dim = 7;
    mjs_setString(bring_sensor->name, "Bring");
    mjs_setDouble(bring_sensor->userdata, (double[]){2 /*L2 norm*/, 1, 0, 1, 0.003}, 5);

#if MJPC_LSQP_SPAWN_OBJECT
    // 9.3- Object sensor
    mjsSensor* obj_pos_sensor = mjs_addSensor(scene_spec);
    obj_pos_sensor->type = mjSENS_FRAMEPOS;
    obj_pos_sensor->objtype = mjOBJ_BODY;
    mjs_setString(obj_pos_sensor->name, (std::string(TARGET_OBJ_NAME) + "_pos").c_str());
    mjs_setString(obj_pos_sensor->objname, TARGET_OBJ_NAME);

    mjsSensor* obj_quat_sensor = mjs_addSensor(scene_spec);
    obj_quat_sensor->type = mjSENS_FRAMEQUAT;
    obj_quat_sensor->objtype = mjOBJ_BODY;
    mjs_setString(obj_quat_sensor->name, (std::string(TARGET_OBJ_NAME) + "_quat").c_str());
    mjs_setString(obj_quat_sensor->objname, TARGET_OBJ_NAME);
#endif

    // 9.4- Palm sensor
    mjsSensor* palm_sensor = mjs_addSensor(scene_spec);
    const auto palm_site_name = PalmSiteName();
    palm_sensor->type = mjSENS_FRAMEPOS;
    palm_sensor->objtype = mjOBJ_SITE;
    mjs_setString(palm_sensor->name, (palm_site_name + "_pos").c_str());
    mjs_setString(palm_sensor->objname, palm_site_name.c_str());

    // 9.5- Fingertip sensors
    int trace_i = 0;
    for (const auto& fingertip : FINGERTIP_NAMES) {
      const auto fingertip_name = attach_prefix_ + fingertip;

      // Trace
      mjsSensor* trace_sensor = mjs_addSensor(scene_spec);
      trace_sensor->type = mjSENS_FRAMEPOS;
      trace_sensor->objtype = mjOBJ_SITE;
      mjs_setString(trace_sensor->name, (TRACE_SENSOR_PREFIX + std::to_string(trace_i++)).c_str());
      mjs_setString(trace_sensor->objname, fingertip_name.c_str());

      // Framepos
      mjsSensor* framepos_sensor = mjs_addSensor(scene_spec);
      framepos_sensor->type = mjSENS_FRAMEPOS;
      framepos_sensor->objtype = mjOBJ_SITE;
      mjs_setString(framepos_sensor->name, (fingertip_name + "_pos").c_str());
      mjs_setString(framepos_sensor->objname, fingertip_name.c_str());

      // Framelinvel
      mjsSensor* framelinvel_sensor = mjs_addSensor(scene_spec);
      framelinvel_sensor->type = mjSENS_FRAMELINVEL;
      framelinvel_sensor->objtype = mjOBJ_SITE;
      mjs_setString(framelinvel_sensor->name, (fingertip_name + "_lin_vel").c_str());
      mjs_setString(framelinvel_sensor->objname, fingertip_name.c_str());

      // Framelinacc
      mjsSensor* framelinacc_sensor = mjs_addSensor(scene_spec);
      framelinacc_sensor->type = mjSENS_FRAMELINACC;
      framelinacc_sensor->objtype = mjOBJ_SITE;
      mjs_setString(framelinacc_sensor->name, (fingertip_name + "_lin_acc").c_str());
      mjs_setString(framelinacc_sensor->objname, fingertip_name.c_str());

      // Frameangvel
      mjsSensor* frameangvel_sensor = mjs_addSensor(scene_spec);
      frameangvel_sensor->type = mjSENS_FRAMEANGVEL;
      frameangvel_sensor->objtype = mjOBJ_SITE;
      mjs_setString(frameangvel_sensor->name, (fingertip_name + "_ang_vel").c_str());
      mjs_setString(frameangvel_sensor->objname, fingertip_name.c_str());

      // Frameangacc
      mjsSensor* frameangacc_sensor = mjs_addSensor(scene_spec);
      frameangacc_sensor->type = mjSENS_FRAMEANGACC;
      frameangacc_sensor->objtype = mjOBJ_SITE;
      mjs_setString(frameangacc_sensor->name, (fingertip_name + "_ang_acc").c_str());
      mjs_setString(frameangacc_sensor->objname, fingertip_name.c_str());
    }

#if MJPC_LSQP_SPAWN_OBJECT
    // 10- Picked object
    mjsBody* pick_obj = mjs_addBody(world_body, nullptr);
    mjs_addFreeJoint(pick_obj);
    mjs_setString(pick_obj->name, TARGET_OBJ_NAME);
    memcpy(pick_obj->pos, TARGET_OBJ_QPOS.data(), sizeof(pick_obj->pos));
    memcpy(pick_obj->quat, TARGET_OBJ_QPOS.data() + 3, sizeof(pick_obj->quat));
    pick_obj->mocap = false;

    mjsGeom* pick_obj_geom = mjs_addGeom(pick_obj, nullptr);
    pick_obj_geom->type = mjGEOM_BOX;
#if 1
    pick_obj_geom->density = 5000000;
#else
    pick_obj->mass = 1;
    memcpy(pick_obj->inertia, (double[]){1., 1., 1.}, sizeof(pick_obj->inertia));
#endif
    memcpy(pick_obj_geom->size, (double[]){0.03, 0.03, 0.03}, sizeof(pick_obj_geom->size));
    memcpy(pick_obj_geom->rgba, (float[]){0.2, 0.5, 0.3, 0.5}, sizeof(pick_obj_geom->rgba));

    // 10.1- Picked obj's target site (!NOTE: Enable site group for visualization)
    auto* target_site = fCreateSite(world_body, TARGET_OBJ_GOAL_NAME, mjGEOM_BOX, 0.03);
    memcpy(target_site->pos, (mjtNum[]){0.1, 0.5, 0.5}, sizeof(target_site->pos));
    memcpy(target_site->quat, (mjtNum[]){0.7, 0., 0.7, 0}, sizeof(target_site->quat));
    memcpy(target_site->rgba, (float[]){0.5, 0., 0., 0.5}, sizeof(target_site->rgba));
#endif

    // 11- Compile [scene_spec] -> mjModel
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
    model->opt.timestep = 0.01;
    // Gravity. NOTE: Enable "gravcomp='1'" in XML for gravity compensation
    model->opt.gravity[2] = KGRAVITY;
#if 0
    // THIS DOES NOT HAVE ANY EFFECT, NEED TO CONFIG mjCBody directly, which is not supported by MuJoCo
    if (KGRAVITY != 0.0) {
      // Enable gravity compensation
      for (auto i = 0; i < model->nbody; ++i) {
        model->body_gravcomp[i] = true;
      }
    }
#endif
  }

  void InitMocaps(bool force_init = false) {
    if (mocaps_inited_ && !force_init) {
      return;
    }
#if MJPC_LSQP_PLANAR_ROBOT
    MoveBodyMocapToSite("target_mocap", "hand");
#else
    mju_copy3(initial_ee_target_pos, QuerySitePos(ATTACHMENT_SITE_NAME));
    MoveBodyMocapToSite(EE_TARGET_NAME, ATTACHMENT_SITE_NAME);
    for (const auto& fingertip_name : FINGERTIP_NAMES) {
      const auto finger_site_name = FingertipSiteName(fingertip_name);
      if (const auto finger_site_id = QuerySiteId(finger_site_name.c_str())) {
        MoveBodyMocapToSite(FingertipTargetBodyName(fingertip_name).c_str(), finger_site_id);
      }
    }

    // Mid position of {palm, fingertips}
    double* palm_pos = QuerySitePos(PalmSiteName().data());
    for (const auto& fingertip : FINGERTIP_NAMES) {
      double* fingertip_pos = QuerySitePos(FingertipSiteName(fingertip).data());
      mju_addTo3(palm_pos, fingertip_pos);
      mju_copy3(initial_fingertips_direction[fingertip], fingertip_pos);
    }
    mju_scl3(palm_pos, palm_pos, 1.0 / (FINGERTIP_NAMES.size() + 1));

    for (const auto& fingertip : FINGERTIP_NAMES) {
      mju_subFrom3(initial_fingertips_direction[fingertip], palm_pos);
      mju_normalize3(initial_fingertips_direction[fingertip]);
    }
#endif
    mocaps_inited_ = true;
  }

  std::vector<double> GetHandCenterPos(const mjData* data) const {
    std::vector<double> center_pos(3, 0);
    // Mid position of {palm, fingertips}
    double* palm_pos = mjpc::QuerySitePos(model_, data, PalmSiteName().data());
    for (const auto& fingertip : FINGERTIP_NAMES) {
      double* fingertip_pos = mjpc::QuerySitePos(model_, data, FingertipSiteName(fingertip).data());
      mju_addTo3(palm_pos, fingertip_pos);
    }
    mju_scl3(center_pos.data(), palm_pos, 0.2);
    return center_pos;
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

public:
  std::vector<double> system_qpos_home;
  double initial_ee_target_pos[3];
  std::map<std::string, double[3]> initial_fingertips_direction; // toward palm

private:
  LsqpPlanner* lsqp_planner_ = nullptr;
  ResidualFn residual_;
  bool mocaps_inited_ = false;
  std::unique_ptr<mjVFS> vfs_ = std::make_unique<mjVFS>();
  std::map<std::string, mjSpec*> spec_map_;
  std::string attach_prefix_;
  std::string attach_suffix_;
};
} // namespace mjpc
