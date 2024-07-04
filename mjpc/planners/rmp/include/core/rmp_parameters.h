#ifndef RMP_PLANNER_PARAMETERS_H
#define RMP_PLANNER_PARAMETERS_H

#include <shared_mutex>
#include <memory>

#define RMP_ISPC (0)
#define RMP_USE_LINEAR_GEOMETRY (1)
#define RMP_USE_RMP_COLLISION_POLICY (1)
#define RMP_USE_ACTUATOR_VELOCITY (1)
#define RMP_USE_ACTUATOR_MOTOR (!RMP_USE_ACTUATOR_VELOCITY)
#define RMP_BLOCKING_OBSTACLES_RATIO (0.1)
#define RMP_BLOCKING_OBSTACLES_SIZE_SCALE (1.2)

#define RMP_COLLISION_USE_3D_TRACE_RAYS (0)
#define RMP_COLLISION_ACTIVE_RADIUS (1.5)
#define RMP_COLLISION_DISTANCE_TRACE_RAYS_NUM (2000)

#define RMP_DRAW_START_GOAL (0)
#define RMP_DRAW_VELOCITY (0)
#define RMP_DRAW_TRAJECTORY (1)
#define RMP_DRAW_BLOCKING_TRACE_RAYS (0)
#define RMP_DRAW_DISTANCE_TRACE_RAYS (1)

#define RMP_KV (0.6)
#define RMP_FORCE_GAIN (1)

using RMPSharedMutexLock = std::shared_lock<std::shared_mutex>;

/**
 * Most of the default values here all get overridden by the parser class.
 */

enum ERMPPolicyType { SIMPLE_ESDF, RAYCASTING };

struct RMPBasePolicyConfigs {
  // For dynamic_cast
  virtual ~RMPBasePolicyConfigs() = default;
};

/** TODO: Fix the redundancy between some of these */
struct ESDFPolicyConfigs : RMPBasePolicyConfigs {
  /** Simple ESDF policy parameters. */
  double eta_repulsive = 22.0; // Gets multiplied by a gain factor from the parser
  double eta_damp = 35.0; // Gets multiplied by a gain factor from the parser
  double v_repulsive = 2.0;
  double v_damp = 2.0;
  double epsilon_damp = 0.1;
  double alpha = 0.2;
  double radius = 5.0; // weighting factor for the softmax
};

struct RaycastingPolicyConfigs : RMPBasePolicyConfigs {
  double eta_repulsive = 0.005; // Gets multiplied by a gain factor from the parser
  double eta_damp = 0.005; // Gets multiplied by a gain factor from the parser
  double v_repulsive = 0.005;
  double v_damp = 0.01;
  double epsilon_damp = 0.1; // [0,1] for numerical stability
  double alpha = 5.0; // weighting factor for the softmax
  double radius = RMP_COLLISION_ACTIVE_RADIUS;
  bool metric = true; // Whether or not using Directionally stretched metrics

#if 0 // UNUSED
  double lin_rep = 1.0;

  double alpha_goal = 10;
  double beta_goal = 20;
  double gamma_goal = 0.02;
  double metric_goal = 1.0;

  double a_fsp = 1e-4;
  double eta_fsp = 0;

  double surface_distance_epsilon_vox = 0.1;
  int N_sqrt = 32;  // square root of number of rays. Has to be divisible by
                    // blocksize TODO: deal with non divisibility
  int max_steps = 100;
#endif
  double truncation_distance_vox = 1.0;
};

struct RMPConfigs {
  RMPConfigs() = default;

  RMPConfigs(const ERMPPolicyType T) : policy_type(T) {
    switch (T) {
      case ERMPPolicyType::SIMPLE_ESDF:
        policyConfigs = std::make_shared<ESDFPolicyConfigs>();
      case ERMPPolicyType::RAYCASTING:
        policyConfigs = std::make_shared<RaycastingPolicyConfigs>();
    }
  }

  ERMPPolicyType policy_type = ERMPPolicyType::RAYCASTING;
  double dt = 0.01;

  // These 2 essentially are subject to each task specifics
  float max_length = 1;
  long max_steps = 1000000;

#if 0 // UNUSED
  bool terminate_upon_goal_reached = true;

  /** Target policy parameters */
  double alpha_target = 10.0;
  double beta_target = 15.0;
  double c_softmax_target = 0.2;
#endif

  std::shared_ptr<RMPBasePolicyConfigs> policyConfigs = nullptr;
};

#endif  // RMP_PLANNER_PARAMETERS_H
