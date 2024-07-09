#include "mjpc/planners/rmp/include/policies/rmp_raycasting_policy.h"

#include <Eigen/QR>
#include <cmath>

// ISPC
#include "rt_ispc.h"

// MuJoCo
#include "mujoco/mujoco.h"

// MJPC
#include "mjpc/utilities.h"
#include "mjpc/planners/rmp/include/core/rmp_space.h"
#include "mjpc/planners/rmp/include/planner/rmp_base_planner.h"

/**
 * Generates a value between 0 and 1 according to the halton sequence
 * @param index ID of the number
 * @param base Base for the halton sequence
 * @return
 */
template <typename TFloat>
inline TFloat halton_seq(int index, int base) {
  TFloat f = 1, r = 0;
  while (index > 0) {
    f = f / base;
    r = r + f * (index % base);
    index = index / base;
  }
  return r;
}

template <typename TFloat>
inline TFloat alpha_freespace(const TFloat d, const TFloat eta_fsp) {
  return eta_fsp * 1.0 / (1.0 + exp(-(2 * d - 6)));
}

/**
 * RMP Obstacle policy repulsive term activation function as defined in the RMP
 * paper
 * @param d Distance to obstacle
 * @param eta_repulsive
 * @param v_repulsive
 * @return
 */
template <typename TFloat>
inline TFloat alpha_repulsive(const TFloat d, const TFloat eta_repulsive,
                              const TFloat v_repulsive,
                              const TFloat linear = 0.0) {
  return eta_repulsive * (exp(-d / v_repulsive)) + (linear * 1 / d);
}

/**
 * RMP Obstacle policy damping term activation function as defined in the RMP
 * paper
 * @param d Distance to obstacle
 * @param eta_damp
 * @param v_damp
 * @param epsilon_damp
 * @return
 */
template <typename TFloat>
inline TFloat alpha_damp(const TFloat d, const TFloat eta_damp,
                         const TFloat v_damp,
                         const TFloat epsilon_damp) {
  return eta_damp / (d / v_damp + epsilon_damp);
}

/**
 * RMP Obstacle policy metric weighing term as defined in the RMP paper
 * @param distance Distance to obstacle
 * @param radius Radius in which the policy is applied/active
 * @return
 */
template <typename TFloat>
inline TFloat obstacle_weight(const TFloat distance, const TFloat radius) {
  const auto& d = distance;
  const auto& r = radius;

  // Disregard obstacles outside of active-scanning zone
  if (d > r) {
    return 0.0f;
  }
#if 1
  // Cubic clamped spline with derivative as 0 at two ends
  return (1.0f / (r * r)) * (d * d * d) - (2.0f / r) * d * d + 1.0f;
#else
  return (1.0f / (r * r)) * (d * d) - (2.0f / r) * d + 1.0f;
#endif
}

/********************************************************************
 ****** Ray tracing kernel code (parts adapted from nvblox)
 ********************************************************************/
template <typename TFloat>
inline std::pair<TFloat, TFloat> get_angles(const TFloat u,
                                            const TFloat v) {
  // Convert uniform sample idx/dimx and idy/dimy to uniform sample on sphere
  float phi = acos(1 - 2 * u);
  float theta = 2.0f * M_PI * v;
  return {phi, theta};
}

template <class TSpace>
std::pair<mjtNum, typename rmp::RaycastingPolicy<TSpace>::Vector>
rmp::RaycastingPolicy<TSpace>::raycastKernel(int ray_id,
  const Vector& ray_start, int target_geomtype,
  const mjtNum* target_pos, const mjtNum* target_rot, const mjtNum* target_size)
{
  // Generate halton sequence and get angles
  const auto u = halton_seq<double>(ray_id, 2);
  const auto v = halton_seq<double>(ray_id, 3);
  const auto angles = get_angles<double>(u, v);
  double phi = angles.first;
  double theta = angles.second;

#if RMP_COLLISION_USE_3D_TRACE_RAYS
  // Convert to direction, of which the meaning itself is already an unit vector
  const mjtNum unit_direction[3] = {
    sin(phi) * cos(theta),
    sin(phi) * sin(theta),
    cos(phi)
  };
#else
  const mjtNum unit_direction[3] = {
    (ray_id %2 == 0) ? sin(phi) : cos(phi),
    (ray_id %3 == 0) ? cos(phi) : sin(phi),
    0.0
  };
#endif
  // https://omaraflak.medium.com/ray-tracing-from-scratch-in-python-41670e6a96f9
#if RMP_ISPC
  const mjtNum distance = ispc::raySphere(target_pos, target_size[0] * target_size[0],
                                          ray_start.data(), unit_direction);
#else
  const mjtNum distance = mju_rayGeom(target_pos, target_rot, target_size,
                                      ray_start.data(), unit_direction,
                                      target_geomtype);
#endif
  Vector ray_unit_direction;
  mju_copy3(ray_unit_direction.data(), unit_direction);
  return {distance, ray_unit_direction};
}

/********************************************************************/
template <class TSpace>
void rmp::RaycastingPolicy<TSpace>::startEval(const PState& agent_state,
                                                     const std::vector<PState>& obstacle_states) {
  // Shoot rays from agent toward obstacles
#pragma omp parallel for if MJPC_OPENMP_ENABLED
  for (auto i = 1; i < RMP_COLLISION_DISTANCE_TRACE_RAYS_NUM; ++i) {
    mjtNum distance_min = std::numeric_limits<mjtNum>::max();
    Vector ray_direction = Vector::Zero();
    // Get shortest distance to obstacles
    for (const auto& obstacle: obstacle_states) {
      const auto _ = raycastKernel(i, agent_state.pos_, mjGEOM_SPHERE,
                                   obstacle.pos_.data(), obstacle.rot_.data(), obstacle.size_.data());
      const auto distance = _.first;
      if ((distance != -1) && (distance < distance_min)) {
        distance_min = distance;
        ray_direction = _.second; // This is already guaranteed a unit direction vector
      }
    }

    // Calculate {f_obs (acceleration), A_metric}
    Vector f_obs = Vector::Zero();
    Matrix A_metric = Matrix::Zero();

    // && (distance_min < RMP_COLLISION_ACTIVE_RADIUS)
    if ((distance_min > 0) && (distance_min != std::numeric_limits<mjtNum>::max())) {
#if RMP_DRAW_DISTANCE_TRACE_RAYS
#pragma omp critical
      {
        this->raytraces_.push_back({.ray_start = agent_state.pos_,
                                    .ray_end = agent_state.pos_ + ray_direction * distance_min,
                                    .distance = distance_min});
      }
#endif
      // Calculate resulting RMP for this target obstacle
      // Unit vector pointing away from the obstacle
      const Vector delta_d = -ray_direction;

      // Simple RMP obstacle policy
      const Vector f_repulsive =
          alpha_repulsive(distance_min, parameters_.eta_repulsive, parameters_.v_repulsive, 0.0) *
                          delta_d;
      // A directionally-scaled projection of [agent_state.vel_] onto [ray_direction],
      // scaled by a factor that vanishes as [agent_state.vel_] moves toward the half space:
      // Haway = {v | delta_d.transpose() * v >= 0}, as orthogonal to or pointing away from the obstacle
      const Vector p_obs = fmax(0.0, double(-agent_state.vel_.transpose() * delta_d)) *
                           (delta_d * delta_d.transpose()) * agent_state.vel_;
      // Original: -alpha_damp
      const Vector f_damp = alpha_damp(distance_min, parameters_.eta_damp,
                                       parameters_.v_damp, parameters_.epsilon_damp) * p_obs;
      f_obs = f_repulsive + f_damp;

      // Obstacle metric
      if (parameters_.metric) {
        // Directionally (f_obs) stretched metric
        const Vector f_norm_metric = this->soft_norm(f_obs, parameters_.alpha);
        // This metric smoothly transitions from [f_norm_metric], stretching along a desired acceleration vector [f_obs],
        // and an uniformed metric [softmax], while being modulated by [parameters_.alpha]
        const auto A_stretch_metric = f_norm_metric * f_norm_metric.transpose();
        A_metric = obstacle_weight(distance_min, parameters_.radius) * A_stretch_metric;
      } else {
        A_metric = obstacle_weight(distance_min, parameters_.radius) * Matrix::Identity();
      }

      // [metric_sum, metric_x_force_sum_]
#pragma omp critical
      {
        //A[0] = A[0] / float(blockdim * dimy);  // scale with number of rays
        metric_sum_.push_back(A_metric);
        metric_x_force_sum_.push_back(A_metric * f_obs);
      }
    } // End if distance_min is valid
  } // End rayshooting loop
}

/**
 * Not implemented for 2d
 * @param state
 * @return
 */
template <>
rmp::RaycastingPolicy<rmp::Space<2>>::PValue
rmp::RaycastingPolicy<rmp::Space<2>>::evaluateAt(
    const PState &state, const std::vector<PState>&) {
  throw std::logic_error("Not implemented");
}

/**
 * Blocking call to evaluate at state.
 * @param state
 * @return
 */
template <>
rmp::RaycastingPolicy<rmp::CylindricalSpace>::PValue
rmp::RaycastingPolicy<rmp::CylindricalSpace>::evaluateAt(
    const PState& agent_state, const std::vector<PState>& obstacle_states) {
  if (!async_eval_started_) {
    startEval(agent_state, obstacle_states);
  }
  /** If an asynchronous eval was started, no check is done whether the state is
   * the same. (As for now this should never happen)*/
  async_eval_started_ = false;

  Matrix sum = Matrix::Zero();
  Vector sumv = Vector::Zero();
  for (int i = 0; i < metric_sum_.size(); ++i) {
    sum += metric_sum_[i];
    sumv += metric_x_force_sum_[i];
  }
  if (sum.isZero(0.001)) {  // Check if not all values are 0, leading to
                            // unstable inverse
    return {Vector::Zero(), Matrix::Zero()};
  }

  Matrix sum_inverse = sum.completeOrthogonalDecomposition().pseudoInverse();
  Vector f = sum_inverse * sumv;
  last_evaluated_state_.pos_ = agent_state.pos_;
  last_evaluated_state_.vel_ = agent_state.vel_;

  return {f, sum};
}

/**
 * Abort asynchronous evaluation
 * @tparam TSpace
 */
template <class TSpace>
void rmp::RaycastingPolicy<TSpace>::abortEvaluateAsync() {
  async_eval_started_ = false;
}

template
void rmp::RaycastingPolicy<rmp::Space<2>>::abortEvaluateAsync();

template
void rmp::RaycastingPolicy<rmp::CylindricalSpace>::abortEvaluateAsync();

template
void rmp::RaycastingPolicy<rmp::Space<3>>::abortEvaluateAsync();
