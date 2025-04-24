#pragma once

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/tasks/lsqp/lsqp.h"

namespace mjpc {
class IIWA14Allegro : public Lsqp {
public:
  using LsqpSolverPtr = Lsqp::LsqpSolverPtr;
  static constexpr const char* BASE_LINK_NAME = "base";
  static constexpr const char* EE_TARGET_NAME = "ee_target";
  static constexpr double EE_TARGET_PREGRASP_QUAT[4] = {0.644963, -0.112683, 0.75546, 0.0246038};
  static constexpr const char* ATTACHMENT_SITE_NAME = "attachment_site";
  static constexpr const char* PALM_NAME = "palm";
  static const std::vector<std::string> FINGERTIP_NAMES;
  static const std::map<std::string, std::vector<float>> FINGERTIPS_RGBA;
  static const std::vector<const char*> FINGERS_JOINT_NAMES;
  static const std::vector<const char*> FINGERS_ACTUATOR_NAMES;
  // NOTE: Hardcoded trace prefix as required in task.cc
  static constexpr const char* TRACE_SENSOR_PREFIX = "trace";

  std::string Name() const override { return "IIWA14Allegro"; }

  std::string XmlPath() const override {
    // Model is manually composed by [ConstructModel()]
    return {};
  }

  IIWA14Allegro() : residual_(this) {
  }

  void TransitionLocked(mjModel* model, mjData* data) override;

  std::string AttachmentPrefix() const { return attach_prefix_; }
  std::string AttachmentSuffix() const { return attach_suffix_; }

  std::string PalmBodyName() const {
    return attach_prefix_ + PALM_NAME;
  }

  std::string PalmSiteName() const {
    return PalmBodyName();
  }

  std::string FingerJointName(const std::string& finger_joint_name) const {
    return attach_prefix_ + finger_joint_name;
  }

  std::string FingerActuatorName(const std::string& finger_act_name) const {
    return attach_prefix_ + finger_act_name;
  }

  std::string FingertipBodyName(const std::string& fingertip_name) const {
    return attach_prefix_ + fingertip_name;
  }

  std::string FingertipSiteName(const std::string& fingertip_name) const {
    return FingertipBodyName(fingertip_name);
  }

  std::string FingertipTargetMocapName(const std::string& fingertip_name) const {
    return FingertipSiteName(fingertip_name) + "_target";
  }

  std::string EETargetName() const override {
    return attach_prefix_ + EE_TARGET_NAME;
  }

  std::string EETargetSiteName() const override {
    return EETargetName();
  }

  std::string EETargetMocapName() const override {
    return EETargetName();
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
    scene_spec->memory = 15000000000;
    scene_spec->nuserdata = LsqpSE3::PARAMS_DIM; // Storing latest [T_wrist] for LsqpPlanner's Diff-IK Control
    scene_spec->option.noslip_iterations = 5;
    //scene_spec->option.noslip_tolerance = 1e-06;
    if (!(scene_spec->option.enableflags & mjENBL_MULTICCD)) {
      scene_spec->option.enableflags |= mjENBL_MULTICCD;
    }

    // 2- Create [allegro_spec]
    auto* allegro_spec = CreateSpecFromXML(allegro_name, ALLEGRO_HAND_XML);
    if (!allegro_spec) { return nullptr; }
    const auto allegro_model_name = std::string(mjs_getString(allegro_spec->modelname));
    mjsBody* allegro_palm = mjs_findBody(allegro_spec, "palm");
    mju_copy4(allegro_palm->quat, mjpc::QUAT_IDENTITY);
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
    fCreateNumeric("agent_timestep", INTEGRATION_DT);
    fCreateNumeric("sampling_trajectories", 10);

    // 4- Attach [allegro_palm] -> [scene_spec] through [attach_site]
    // NOTE: This prefix will be prepended to names of all child elements (bodies, geoms, etc.) in [allegro_model]
    attach_prefix_ = allegro_model_name + "/";
    mjsSite* attach_site = mjpc::FindSiteSpec(scene_spec, ATTACHMENT_SITE_NAME);
    mjs_attach(attach_site->element, allegro_palm->element, attach_prefix_.c_str(), attach_suffix_.c_str());

    // 5- Customize joints
    for (const auto& jnt_name : FINGERS_JOINT_NAMES) {
      auto* jnt_spec = mjs_asJoint(mjs_findElement(scene_spec, mjOBJ_JOINT,
                                                   FingerJointName(jnt_name).c_str()));
      jnt_spec->armature = 0.05;
    }

    // 6.1- Customize [IIWA14] actuactors
    if constexpr (SYSTEM_MODEL_ACTUATORS_OSC) {
      // Delete all the default fully-actuated actuators
      for (auto i = 1; i <= IIWA14_DOF; i++) {
        auto* act_spec = mjs_findElement(scene_spec, mjOBJ_ACTUATOR,
                                         ("actuator" + std::to_string(i)).c_str());
        if (mjs_asActuator(act_spec)) {
          mjs_delete(act_spec);
        }
      }

      for (const auto& act_name : FINGERS_ACTUATOR_NAMES) {
        auto* act_spec = mjs_findElement(allegro_spec, mjOBJ_ACTUATOR, act_name);
        if (mjs_asActuator(act_spec)) {
          mjs_delete(act_spec);
        }
      }
    } else {
      for (auto i = 1; i <= IIWA14_DOF; i++) {
        auto* act = mjs_asActuator(mjs_findElement(scene_spec, mjOBJ_ACTUATOR,
                                                   ("actuator" + std::to_string(i)).c_str()));
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

    // 6.2- Customize [ALLEGRO] actuators
    for (const auto& act_name : FINGERS_ACTUATOR_NAMES) {
      auto* act = mjs_asActuator(mjs_findElement(scene_spec, mjOBJ_ACTUATOR,
                                                 FingerActuatorName(act_name).c_str()));
      act->trntype = mjTRN_JOINT;
      act->dyntype = mjDYN_NONE;
      act->biastype = mjBIAS_AFFINE;
      act->gaintype = mjGAIN_FIXED;
      constexpr double kp = 10;
      constexpr double kv = 1;
      if constexpr (POSITION_CTRL_ENABLED) {
        mju_copy3(act->gainprm, (double[]){kp, 0, 0});
        mju_copy3(act->biasprm, (double[]){0, -kp, -kv});
      } else {
        mju_copy3(act->gainprm, (double[]){kv, 0, 0});
        mju_copy3(act->biasprm, (double[]){0, 0, -kv});
      }
    }

    // 7- Re-key new assembled robot model [scene_spec]
    auto* key_home = mjpc::FindKeySpec(scene_spec, "home");
    if (!key_home) {
      key_home = mjs_addKey(scene_spec);
      mjs_setString(key_home->name, "home");
    }

    // 7.1- Key qpos
#if MJPC_LSQP_SPAWN_OBJECT
    system_qpos_home_ = mjpc::ChainCollections<mjtNum>(IIWA14_ALLEGRO_HOME_QPOS,
                                                       mjpc::CollectionFromCArray(TARGET_OBJ_QPOS));
#else
    system_qpos_home_ = IIWA14_ALLEGRO_HOME_QPOS;
#endif
    mjs_setDouble(key_home->qpos, system_qpos_home_.data(), system_qpos_home_.size());

    // 7.2- Key ctrl
    if constexpr (SYSTEM_MODEL_ACTUATORS_OSC) {
    } else {
      mjs_setDouble(key_home->ctrl, IIWA14_ALLEGRO_HOME_QPOS.data(), IIWA14_ALLEGRO_HOME_QPOS.size());
    }

    // 7- Add mocap bodies
    mjsBody* world_body = mjpc::FindWorldBodySpec(scene_spec);
    const auto fCreateSite = [](mjsBody* body, const std::string& site_name,
                                double pos[3] = nullptr, double quat[4] = nullptr,
                                mjtGeom type = mjGEOM_SPHERE, double size = 0.001,
                                float rgba[4] = nullptr) {
      mjsSite* site = mjs_addSite(body, nullptr);
      mjs_setString(site->name, site_name.c_str());
      site->type = type;
      memcpy(site->size, (mjtNum[]){size, size, size}, sizeof(site->size));
      if (pos) { memcpy(site->pos, pos, sizeof(site->pos)); }
      if (quat) { memcpy(site->quat, quat, sizeof(site->quat)); }
      if (rgba) { memcpy(site->rgba, (float[]){0.5, 0., 0., 0.5}, sizeof(site->rgba)); }
      site->group = 4;
      return site;
    };

    // 7.1- [ee-mocap body]
    mjsBody* ee_mocap = mjs_addBody(world_body, nullptr);
    mjs_setString(ee_mocap->name, EETargetMocapName().data());
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
      fCreateSite(ee_mocap, EETargetSiteName());
    } else {
      fCreateSite(allegro_palm, EETargetSiteName(), (double[]){0.03, 0, 0.03});
      fCreateSite(allegro_palm, PalmSiteName());
    }

    // 7.2- [Fingertip-mocap bodies]
    for (const auto& fingertip : FINGERTIP_NAMES) {
      const auto fingertip_target_name = FingertipTargetMocapName(fingertip);
      mjsBody* finger_mocap = mjs_addBody(world_body, nullptr);
      mjs_setString(finger_mocap->name, fingertip_target_name.c_str());
      finger_mocap->mocap = true;
      mjsGeom* finger_mocap_geom = mjs_addGeom(finger_mocap, nullptr);
      finger_mocap_geom->type = mjGEOM_SPHERE;
      memcpy(finger_mocap_geom->size, (mjtNum[]){0.02, 0.02, 0.02}, sizeof(finger_mocap_geom->size));
      memcpy(finger_mocap_geom->rgba, FINGERTIPS_RGBA.at(fingertip).data(),
             sizeof(finger_mocap_geom->rgba));
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
      const auto eetarget_site_name = EETargetSiteName();
      fCreateActuator("ee_act_x", eetarget_site_name, (double[]){1, 0, 0, 0, 0, 0});
      fCreateActuator("ee_act_y", eetarget_site_name, (double[]){0, 1, 0, 0, 0, 0});
      fCreateActuator("ee_act_z", eetarget_site_name, (double[]){0, 0, 1, 0, 0, 0});
      for (const auto& fingertip : FINGERTIP_NAMES) {
        const auto fingertip_mocap_site_name = FingertipTargetMocapName(fingertip);
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

    // 10- EE Target sensor
    mjsSensor* ee_target_sensor = mjs_addSensor(scene_spec);
    const auto ee_target_site_name = EETargetSiteName();
    ee_target_sensor->type = mjSENS_FRAMEPOS;
    ee_target_sensor->objtype = mjOBJ_SITE;
    mjs_setString(ee_target_sensor->name, (ee_target_site_name + "_pos").c_str());
    mjs_setString(ee_target_sensor->objname, ee_target_site_name.c_str());

#if MJPC_LSQP_SPAWN_OBJECT
    // 11- Picked object
    mjsBody* pick_obj = mjs_addBody(world_body, nullptr);
    mjsJoint* pick_obj_jnt = mjs_addFreeJoint(pick_obj);
    pick_obj_jnt->armature = 0.1;
    pick_obj_jnt->damping = 0.5;
    mjs_setString(pick_obj->name, TARGET_OBJ_NAME);
    memcpy(pick_obj->pos, TARGET_OBJ_QPOS, sizeof(pick_obj->pos));
    memcpy(pick_obj->quat, TARGET_OBJ_QPOS + 3, sizeof(pick_obj->quat));
    pick_obj->mocap = false;

    mjsGeom* pick_obj_geom = mjs_addGeom(pick_obj, nullptr);
    pick_obj_geom->type = mjGEOM_BOX;
#if 1
    pick_obj_geom->density = 8000;
#else
    pick_obj->mass = 1;
    memcpy(pick_obj->inertia, (double[]){1., 1., 1.}, sizeof(pick_obj->inertia));
#endif
    memcpy(pick_obj_geom->size, (double[]){0.03, 0.03, 0.03}, sizeof(pick_obj_geom->size));
    memcpy(pick_obj_geom->rgba, (float[]){0.2, 0.5, 0.3, 0.5}, sizeof(pick_obj_geom->rgba));

    // 11.1- Picked obj's target goal site (!NOTE: Enable site group for visualization)
    fCreateSite(world_body, TARGET_OBJ_GOAL_NAME,
                /*pos*/(mjtNum[]){0.1, 0.5, 0.5},/*quat*/(mjtNum[]){0.7, 0., 0.7, 0},
                mjGEOM_BOX, /*size*/0.03, /*rgba*/(float[]){0.5, 0., 0., 0.5});
#endif

    // 12- Compile [scene_spec] -> mjModel
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

  void InitMocaps() override {
#if MJPC_LSQP_PLANAR_ROBOT
    MoveBodyMocapToSite("target_mocap", "hand");
#else
    const auto ee_site_name = EETargetSiteName();
    mju_copy3(initial_ee_target_pos, QuerySitePos(ee_site_name.data()));
    mju_copy3(initial_ee_target_quat, QuerySiteQuat(ee_site_name.data()));
    MoveBodyMocapToSite(EETargetMocapName().data(), ee_site_name.data());
    for (const auto& fingertip_name : FINGERTIP_NAMES) {
      const auto finger_site_name = FingertipSiteName(fingertip_name);
      if (const auto finger_site_id = QuerySiteId(finger_site_name.c_str())) {
        MoveBodyMocapToSite(FingertipTargetMocapName(fingertip_name).c_str(), finger_site_id);
      }
    }

    for (const auto& fingertip : FINGERTIP_NAMES) {
      mju_sub3(initial_fingertips_direction[fingertip],
               QuerySitePos(FingertipSiteName(fingertip).data()), initial_ee_target_pos);
      mju_normalize3(initial_fingertips_direction[fingertip]);
    }
#endif
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

  double initial_ee_target_pos[3];
  double initial_ee_target_quat[4];
  std::map<std::string, double[3]> initial_fingertips_direction; // toward palm

  double visual_policy_ee_target_pos_[3];
  double visual_ee_direction_[3];
  double palm_normal_[3];

  void InitSolverConfigs(const LsqpSolverPtr& solver, const mjData* data, int ndofs) override;
  std::vector<double> Control(double* policy_action, mjData* data,
                              const LsqpSolverPtr& solver) override;
  std::vector<double> Solve(const LsqpSolverPtr& solver, const mjData* data) override;
  std::string attach_prefix_;
  std::string attach_suffix_;

  void DrawTraces() override;

protected:
  class ResidualFn : public BaseResidualFn {
  public:
    explicit ResidualFn(const IIWA14Allegro* task) : BaseResidualFn(task) {
    }

    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }

  ResidualFn* InternalResidual() override { return &residual_; }

private:
  ResidualFn residual_;
};
} // namespace mjpc
