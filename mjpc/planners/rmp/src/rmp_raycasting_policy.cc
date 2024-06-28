#include "mjpc/planners/rmp/include/policies/rmp_raycasting_policy.h"

#include <Eigen/QR>
#include <cmath>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"
#include "mjpc/planners/rmp/include/core/rmp_space.h"
#include "mjpc/planners/rmp/include/planner/rmp_base_planner.h"

#define BLOCK_SIZE 8

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

template <typename TVector, typename TFloat>
inline TVector softnorm(const TVector& v, const TFloat c) {
  TFloat norm = v.norm();
  TFloat h = norm + c * log(1.0f + exp(-2.0f * c * norm));

  return v / h;
}

template <typename TFloat>
inline TFloat alpha_freespace(const TFloat d, const TFloat eta_fsp) {
  return eta_fsp * 1.0 / (1.0 + exp(-(2 * d - 6)));
}

/**
 * RMP Obstacle policy repulsive term activation function as defined in the RMP
 * paper
 * @param d Distance to obstacle
 * @param eta_rep
 * @param v_rep
 * @return
 */
template <typename TFloat>
inline TFloat alpha_rep(const TFloat d, const TFloat eta_rep,
                       const TFloat v_rep,
                       const TFloat linear = 0.0) {
  return eta_rep * (exp(-d / v_rep)) + (linear * 1 / d);
}

/**
 * RMP Obstacle policy damping term activation function as defined in the RMP
 * paper
 * @param d Distance to obstacle
 * @param eta_damp
 * @param v_damp
 * @param epsilon
 * @return
 */
template <typename TFloat>
inline TFloat alpha_damp(const TFloat d, const TFloat eta_damp,
                         const TFloat v_damp,
                         const TFloat epsilon) {
  return eta_damp / (d / v_damp + epsilon);
}

/**
 * RMP Obstacle policy metric weighing term as defined in the RMP paper
 * @param d Distance to obstacle
 * @param r
 * @return
 */
template <typename TFloat>
inline TFloat w(const TFloat d, const TFloat r) {
  if (d > r) {
    return 0.0f;
  }
  return (1.0f / (r * r)) * d * d - (2.0f / r) * d + 1.0f;
}

#if 1
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
std::pair<typename rmpcpp::RaycastingCudaPolicy<TSpace>::Matrix, typename rmpcpp::RaycastingCudaPolicy<TSpace>::Vector>
rmpcpp::RaycastingCudaPolicy<TSpace>::raycastKernel(int ray_id,
  const Vector& ray_start, const Vector& ray_vel, int target_geomtype,
  const mjtNum* target_pos, const mjtNum* target_rot, const mjtNum* target_size)
{
  // Generate halton sequence and get angles
  const auto u = halton_seq<double>(ray_id, 2);
  const auto v = halton_seq<double>(ray_id, 3);
  const auto angles = get_angles<double>(u, v);
  double phi = angles.first;
  double theta = angles.second;

  // Convert to direction
  const mjtNum direction[3] = {
    sin(phi) * cos(theta),
    sin(phi) * sin(theta),
    0.//cos(phi)
  };
  Vector ray_direction;
  mju_copy3(ray_direction.data(), direction);
  const auto distance = mju_rayGeom(target_pos, target_rot, target_size,
                                    ray_start.data(), direction,
                                    target_geomtype);

  Matrix A = Matrix::Zero();
  Vector f_out = Vector::Zero();

  if (distance != -1) {
#if RMP_DRAW_DISTANCE_TRACE_RAYS
    this->raytraces_.push_back({.ray_start = ray_start, .ray_end = ray_start + ray_direction, .distance = distance});
#endif
    // Calculate resulting RMP for this target obstacle
    // Unit vector pointing away from the obstacle
    Vector delta_d = -ray_direction / ray_direction.norm();

    // Simple RMP obstacle policy
    Vector f_rep =
        alpha_rep(distance, parameters_.eta_rep, parameters_.v_rep, 0.0) *
        delta_d;
    Vector f_damp = -alpha_damp(distance, parameters_.eta_damp,
                                parameters_.v_damp, parameters_.epsilon_damp) *
                    fmax(0.0, double(-ray_vel.transpose() * delta_d)) *
                    (delta_d * delta_d.transpose()) * ray_vel;
    f_out = f_rep + f_damp;
    Vector f_norm = softnorm(f_out, parameters_.c_softmax_obstacle);

    if (parameters_.metric) {
      A = w(distance, parameters_.r) * f_norm * f_norm.transpose();
    } else {
      A = w(distance, parameters_.r) * Matrix::Identity();
    }
  }
  return std::pair{std::move(A), std::move(f_out)};
}
#endif

/********************************************************************/
template <class TSpace>
void rmpcpp::RaycastingCudaPolicy<TSpace>::ispcStartEval(const PState& agent_state,
                                                         const std::vector<PState>& obstacle_states) {
  // Shoot rays from agent toward obstacles
  for (const auto& obstacle : obstacle_states) {
    for (auto i = 1; i < RMP_DISTANCE_TRACE_RAYS_NUM; ++i) {
      auto A_f = raycastKernel(i, agent_state.pos_, agent_state.vel_, mjGEOM_SPHERE,
                               obstacle.pos_.data(), obstacle.rot_.data(), obstacle.size_.data());
      //A_f[0] = A_f[0] / float(blockdim * dimy);  // scale with number of rays
      metric_sum_.push_back(A_f.first);
      metric_x_force_sum_.push_back(A_f.first * A_f.second);
    }
  }
}

/**
 * Not implemented for 2d
 * @param state
 * @return
 */
template <>
rmpcpp::RaycastingCudaPolicy<rmpcpp::Space<2>>::PValue
rmpcpp::RaycastingCudaPolicy<rmpcpp::Space<2>>::evaluateAt(
    const PState &state, const std::vector<PState>&) {
  throw std::logic_error("Not implemented");
}

/**
 * Blocking call to evaluate at state.
 * @param state
 * @return
 */
template <>
rmpcpp::RaycastingCudaPolicy<rmpcpp::CylindricalSpace>::PValue
rmpcpp::RaycastingCudaPolicy<rmpcpp::CylindricalSpace>::evaluateAt(
    const PState& agent_state, const std::vector<PState>& obstacle_states) {
  static const int blockdim = parameters_.N_sqrt / BLOCK_SIZE;
  static const int blockdim_2 = blockdim * blockdim;
  if (metric_sum_.empty()) {
    metric_sum_.reserve(blockdim_2);
  }
  else
  {
    metric_sum_.clear();
  }
  if(metric_x_force_sum_.empty()) {
    metric_x_force_sum_.reserve(blockdim_2);
  }
  else {
    metric_x_force_sum_.clear();
  }
  if (!async_eval_started_) {
    for (int i = 0; i < blockdim_2; ++i) {
      ispcStartEval(agent_state, obstacle_states);
    }
  }
  /** If an asynchronous eval was started, no check is done whether the state is
   * the same. (As for now this should never happen)*/
  async_eval_started_ = false;

  Matrix sum = Matrix::Zero();
  Vector sumv = Vector::Zero();
  for (int i = 0; i < blockdim_2; ++i) {
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
void rmpcpp::RaycastingCudaPolicy<TSpace>::abortEvaluateAsync() {
  async_eval_started_ = false;
}

template
void rmpcpp::RaycastingCudaPolicy<rmpcpp::Space<2>>::abortEvaluateAsync();

template
void rmpcpp::RaycastingCudaPolicy<rmpcpp::CylindricalSpace>::abortEvaluateAsync();

template
void rmpcpp::RaycastingCudaPolicy<rmpcpp::Space<3>>::abortEvaluateAsync();
