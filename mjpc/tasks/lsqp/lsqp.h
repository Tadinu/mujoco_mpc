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

#define MJPC_HOME_LAP (0)
#define MJPC_LSQP_PLANAR_ROBOT (0)
#define MJPC_LSQP_SPAWN_OBJECT (1)

// MANUAL CONTROL USING DIFFIK or OSC
#define MJPC_LSQP_MANUAL_MODE (0)
#define MJPC_LSQP_MANUAL_DIFFIK_ENABLED (MJPC_LSQP_MANUAL_MODE && 0)
#define MJPC_LSQP_MANUAL_OSC_ENABLED (MJPC_LSQP_MANUAL_MODE && !MJPC_LSQP_MANUAL_DIFFIK_ENABLED)
#define MJPC_LSQP_FINGERS_OSC (MJPC_LSQP_MANUAL_OSC_ENABLED & 1)

// AUTO CONTROL USING LSQP PLANNER [SOLVER]
#define MJPC_LSQP_AUTO_MODE (!MJPC_LSQP_MANUAL_MODE)

namespace mjpc {
static const std::string MUJOCO_DIR =
#if MJPC_HOME_LAP
    "/home/tad/1_MUJOCO";
#else
    "/media/ducthan/376b23a1-5a02-4960-b3ca-24b2fcef8f891/MUJOCO";
#endif
static const std::string MAIN_ROBOT_MODEL_PATH = MUJOCO_DIR +
#if MJPC_LSQP_MANUAL_DIFFIK_ENABLED
                                                 //"/mink/examples/franka_emika_panda/panda_nohand.xml";
                                                 //"/mink/examples/universal_robots_ur5e/ur5e.xml";
                                                 //"/mujoco_mpc/mjpc/tasks/bimanual/cobring/bi-franka_panda.xml";
                                                 "/mujoco_mpc/mjpc/tasks/garmi/garmi.xml";
#elif MJPC_LSQP_MANUAL_OSC_ENABLED
                                                 //"/mink/examples/kuka_iiwa_14/iiwa14_osc.xml";
                                                 "/mink/examples/kuka_iiwa_14_allegro/iiwa14_allegro_osc.xml";
#else
                                                 "/mink/examples/arm_hand_iiwa_allegro.xml";
#endif

static const std::string MAIN_ROBOT_MODEL_NAME = std::filesystem::path(MAIN_ROBOT_MODEL_PATH).stem();

static bool IsUR5() {
  return absl::StrContainsIgnoreCase(MAIN_ROBOT_MODEL_NAME, "ur5");
}

static bool IsIIWA14() {
  return absl::StrContainsIgnoreCase(MAIN_ROBOT_MODEL_NAME, "iiwa14");
}

static bool IsIIWA14Allegro() {
  return absl::StrContainsIgnoreCase(MAIN_ROBOT_MODEL_NAME, "iiwa14_allegro");
}

static bool IsGarmi() {
  return absl::StrContainsIgnoreCase(MAIN_ROBOT_MODEL_NAME, "garmi");
}

static bool IsPanda() {
  return absl::StrContainsIgnoreCase(MAIN_ROBOT_MODEL_NAME, "panda");
}

static bool IsBiFrankaPanda() {
  return absl::StrContainsIgnoreCase(MAIN_ROBOT_MODEL_NAME, "bi-franka_panda");
}

static bool IsDualPanda() {
  return IsGarmi() || IsBiFrankaPanda();
}

static bool RobotHasNoHand() {
  return absl::StrContainsIgnoreCase(MAIN_ROBOT_MODEL_NAME, "nohand") || IsUR5() || IsIIWA14() || IsPanda();
}

static bool RobotHasHand() {
  return !RobotHasNoHand();
}

// NOTE: Consider using std::array<>, to be more explicit with size info, so less error-prone?
static constexpr mjtNum IIWA14_HOME_QPOS[] = {
    -0.0759329, 0.153982, 0.104381, -1.8971, 0.245996, 0.34972, -0.239115
};

static constexpr mjtNum ALLEGRO_RF_HOME_QPOS[] = {
    -0.0694123, 0.0551428, 0.986832, 0.671424
};
static constexpr uint8_t ALLEGRO_SINGLE_FINGER_QPOS_SIZE = sizeof(ALLEGRO_RF_HOME_QPOS) / sizeof(mjtNum);

static constexpr mjtNum ALLEGRO_MF_HOME_QPOS[] = {
    -0.186261, -0.0866821, 1.01374, 0.728192
};

static constexpr mjtNum ALLEGRO_FF_HOME_QPOS[] = {
    -0.218949, -0.0318307, 1.25156, 0.840648
};

static constexpr mjtNum ALLEGRO_TH_HOME_QPOS[] = {
    1.0593, 0.638801, 0.391599, 0.57284
};

// NOTE: The order must match ones in [ALLEGRO_EE_NAMES] & [ALLEGRO_ACTUATED_JOINT_NAMES]
static const std::vector<const mjtNum*> ALLEGRO_HOME_QPOS = {ALLEGRO_RF_HOME_QPOS, ALLEGRO_MF_HOME_QPOS,
                                                             ALLEGRO_FF_HOME_QPOS, ALLEGRO_TH_HOME_QPOS
};

static const std::vector<mjtNum> IIWA14_ALLEGRO_HOME_QPOS = []() {
  std::vector<mjtNum> res = mjpc::CollectionFromCArray(IIWA14_HOME_QPOS);
  for (const auto& qpos_adr : ALLEGRO_HOME_QPOS) {
    res = mjpc::ChainCollections<mjtNum>(
        res, std::vector(qpos_adr, qpos_adr + ALLEGRO_SINGLE_FINGER_QPOS_SIZE));
  }
  return res;
}();

static constexpr mjtNum TARGET_OBJ_QPOS[] = {0.9, 0, 0.3, 1, 0, 0, 0};

static constexpr uint8_t IIWA14_DOF = 7;
static constexpr uint8_t ALLEGRO_DOF = 16;
static constexpr uint8_t EE_CEM_PARAMS_DIM = 1; // wrist(XYZ-loc ratio away from goal + theta-rot]
static constexpr uint8_t FINGERS_CEM_PARAMS_DIM = 0; //MJPC_LSQP_FINGERS_OSC
//? 4 // Fingertips
//: ALLEGRO_DOF;
static constexpr uint8_t CEM_PARAMS_TOTAL_DIM = EE_CEM_PARAMS_DIM + FINGERS_CEM_PARAMS_DIM;
static constexpr float CEM_PARAMS_LIMIT_LOWER = 0.;
static constexpr float CEM_PARAMS_LIMIT_UPPER = 1.;

static const std::string MAIN_SCENE_XML_PATH =
#if MJPC_LSQP_MANUAL_DIFFIK_ENABLED
    MUJOCO_DIR + (IsUR5()
                    ? "/mjctrl/universal_robots_ur5e/scene.xml"
                    : IsBiFrankaPanda()
                    ? "/mujoco_mpc/mjpc/tasks/bimanual/cobring/task.xml"
                    : IsPanda()
                    ? "/mjctrl/franka_emika_panda/scene.xml"
                    : IsGarmi()
                    ? "/mujoco_mpc/mjpc/tasks/garmi/garmi_scene.xml"
                    : "");
#elif MJPC_LSQP_MANUAL_OSC_ENABLED
    MUJOCO_DIR + (IsIIWA14Allegro()
                    ? "/mink/examples/kuka_iiwa_14_allegro/scene_osc_target.xml"
                    : "/mink/examples/kuka_iiwa_14/scene_osc_target.xml");
#else
    {};
#endif

// NOTE: Empty base body name is considered as world/root body
static constexpr const char* IIWA14_BASE_DOF_BODY_NAME = "link1"; // Lowest body having dof
static const std::vector<std::string> IIWA14_EE_NAMES = {"attachment"};
static const std::vector<std::string> IIWA14_EE_SITE_NAMES = {"attachment_site"};

static const std::vector<std::string> IIWA14_ACTUATED_JOINT_NAMES = {
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7"
};

static constexpr const char* ALLEGRO_BASE_DOF_BODY_NAME = "allegro_left/rf_base"; // Lowest body having dof
static const std::vector<std::string> ALLEGRO_EE_NAMES = {
    "allegro_left/rf_tip",
    "allegro_left/mf_tip",
    "allegro_left/ff_tip",
    "allegro_left/th_tip"
};
static const std::vector<std::string> ALLEGRO_EE_SITE_NAMES = ALLEGRO_EE_NAMES;
static const std::vector<std::vector<std::string>> ALLEGRO_ACTUATED_JOINT_NAMES = {
    {
        "allegro_left/rfj0", "allegro_left/rfj1", "allegro_left/rfj2", "allegro_left/rfj3"
    },
    {
        "allegro_left/mfj0", "allegro_left/mfj1", "allegro_left/mfj2", "allegro_left/mfj3"
    },
    {
        "allegro_left/ffj0", "allegro_left/ffj1", "allegro_left/ffj2", "allegro_left/ffj3"
    },
    {
        "allegro_left/thj0", "allegro_left/thj1", "allegro_left/thj2", "allegro_left/thj3"
    }
};

static const std::vector<std::string> UR5_ACTUATED_JOINT_NAMES = {
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3"
};

static const std::vector<std::string> PANDA_ACTUATED_JOINT_NAMES = {
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7"
};

static const std::vector<std::string> BIFRANKA_ACTUATED_JOINT_NAMES = {
    "panda0_joint1",
    "panda0_joint2",
    "panda0_joint3",
    "panda0_joint4",
    "panda0_joint5",
    "panda0_joint6",
    "panda0_joint7",
    "panda0_joint1",
    "panda1_joint2",
    "panda1_joint3",
    "panda1_joint4",
    "panda1_joint5",
    "panda1_joint6",
    "panda1_joint7"
};

static const std::vector<std::string> BIFRANKA_EE_NAMES = {"panda0_end_effector", "panda1_end_effector"};
static const std::vector<std::string> BIFRANKA_EE_SITE_NAMES = BIFRANKA_EE_NAMES;

static const std::vector<std::string> GARMI_ACTUATED_JOINT_NAMES = {
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7"
};

static const std::vector<std::string> GARMI_ACTUATOR_NAMES = {
#if 0
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7"
#else
    "left_act_pos1",
    "left_act_pos2",
    "left_act_pos3",
    "left_act_pos4",
    "left_act_pos5",
    "left_act_pos6",
    "left_act_pos7",
    "right_act_pos1",
    "right_act_pos2",
    "right_act_pos3",
    "right_act_pos4",
    "right_act_pos5",
    "right_act_pos6",
    "right_act_pos7"
#endif
};

static const std::vector<std::string> GARMI_EE_NAMES = {"left_hand", "right_hand"};
static const std::vector<std::string> GARMI_EE_SITE_NAMES = {"left_ee_site", "right_ee_site"};

static const std::string MAIN_ROBOT_BASE_LINK_NAME = IsDualPanda()
                                                       ? "torso"
                                                       : IsPanda()
                                                       ? "link0"
                                                       : "base";
static std::vector<std::string> MAIN_ROBOT_EE_NAMES = IsIIWA14Allegro()
                                                        ? mjpc::ChainCollections<std::string>(
                                                            IIWA14_EE_NAMES,
                                                            ALLEGRO_EE_NAMES)
                                                        : IsBiFrankaPanda()
                                                        ? BIFRANKA_EE_NAMES
                                                        : IsGarmi()
                                                        ? GARMI_EE_NAMES
                                                        : std::vector<std::string>{
                                                            RobotHasHand()
                                                              ? "hand"
                                                              : "attachment"};

static const std::vector<std::string> MAIN_ROBOT_EE_SITE_NAMES = IsIIWA14Allegro()
                                                                   ? mjpc::ChainCollections<std::string>(
                                                                       IIWA14_EE_SITE_NAMES,
                                                                       ALLEGRO_EE_SITE_NAMES)
                                                                   : IsBiFrankaPanda()
                                                                   ? BIFRANKA_EE_SITE_NAMES
                                                                   : IsGarmi()
                                                                   ? GARMI_EE_SITE_NAMES
                                                                   : std::vector<std::string>{
                                                                       RobotHasHand()
                                                                         ? "hand_site"
                                                                         : "attachment_site"
                                                                   };

static const std::vector<std::string> MAIN_ACTUATED_JOINT_NAMES = mjpc::IsIIWA14()
                                                                    ? mjpc::IIWA14_ACTUATED_JOINT_NAMES
                                                                    : mjpc::IsUR5()
                                                                    ? mjpc::UR5_ACTUATED_JOINT_NAMES
                                                                    : mjpc::IsBiFrankaPanda()
                                                                    ? mjpc::BIFRANKA_ACTUATED_JOINT_NAMES
                                                                    : mjpc::IsGarmi()
                                                                    ? mjpc::GARMI_ACTUATED_JOINT_NAMES
                                                                    : mjpc::IsPanda()
                                                                    ? mjpc::PANDA_ACTUATED_JOINT_NAMES
                                                                    : std::vector<std::string>{};
static const std::vector<std::string> MAIN_ACTUATOR_NAMES = mjpc::IsGarmi()
                                                              ? mjpc::GARMI_ACTUATOR_NAMES
                                                              : std::vector<std::string>{};

static const bool MAIN_ROBOT_NULLSPACE_CONTROL_ENABLED = false; // IsPanda() || IsDualPanda();
// NOTE: Not necessarily the same as [model->opt.timestep]
static const double INTEGRATION_DT =
#if MJPC_LSQP_MANUAL_MODE
    MAIN_ROBOT_NULLSPACE_CONTROL_ENABLED ? 0.1 : 1;
#else
    0.005;
#endif

class LsqpPlanner;
class LsqpSolver;

class Lsqp : public Task {
public:
  using LsqpSolverPtr = std::shared_ptr<LsqpSolver>;
  static constexpr float KGRAVITY = -9.81f;
  static constexpr bool SYSTEM_MODEL_ACTUATORS_OSC = false;
  static constexpr bool POSITION_CTRL_ENABLED = true;

  static constexpr const char* TARGET_OBJ_NAME = "target";
  static constexpr const char* TARGET_OBJ_GOAL_NAME = "target_goal";

  Lsqp() {
    assert(std::ifstream(mjpc::MUJOCO_DIR.c_str()).good());
    mj_defaultVFS(vfs_.get());
    for (const auto& [_, spec] : spec_map_) {
      mj_deleteSpec(spec);
    }
  }

  void SetPlanner(Planner* planner) override;

  bool IsLSQPSupported() const override { return true; }

  virtual std::string EETargetName() const {
    return {};
  }

  virtual std::string EETargetSiteName() const {
    return {};
  }

  virtual std::string EETargetMocapName() const {
    return {};
  }

  void ConfigureModel(mjModel* model) override {
    // Configure [model]
    //model->opt.disableactuator |= (1 << 0);
    model->opt.timestep = INTEGRATION_DT;
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

  virtual void InitMocaps() {
  }

  virtual std::vector<double> Control(double* policy_action, mjData* data = nullptr,
                                      const LsqpSolverPtr& solver = nullptr) {
    return {};
  }

  void InitSolver(const MjOwnerAppType owner_type, int ndofs);

  virtual void InitSolverConfigs(const LsqpSolverPtr& solver, const mjData* data, int ndofs) {
  }

  virtual std::vector<double> Solve(const LsqpSolverPtr& solver, const mjData* data) {
    return {};
  }

  void ResetLocked(const mjModel* model) override {
    Task::ResetLocked(model);
  }

  virtual void DrawTraces() {
  }

protected:
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
  std::vector<double> system_qpos_home_;
  std::shared_ptr<LsqpSolver> lsqp_solver_ = nullptr;

protected:
  LsqpPlanner* lsqp_planner_ = nullptr;
  std::unique_ptr<mjVFS> vfs_ = std::make_unique<mjVFS>();
  std::map<std::string, mjSpec*> spec_map_;
};

using LsqpPtr = std::shared_ptr<Lsqp>;
} // namespace mjpc
