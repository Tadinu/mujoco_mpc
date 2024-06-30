
#ifndef RMPCPP_PLANNER_RAYCASTING_CUDA_H
#define RMPCPP_PLANNER_RAYCASTING_CUDA_H

#include "mjpc/planners/rmp/include/core/rmp_parameters.h"
#include "mujoco/mujoco.h"
#include "rmp_base_policy.h"

#define OUTPUT_RAYS (0)

namespace rmpcpp {

/*
 * Implements a nvblox- map based raycasting
 * obstacle avoidance policy
 */
template <class TSpace>
class RaycastingPolicy : public RMPPolicyBase<TSpace> {
 public:
  using Vector = typename RMPPolicyBase<TSpace>::Vector;
  using Matrix = typename RMPPolicyBase<TSpace>::Matrix;
  using PValue = typename RMPPolicyBase<TSpace>::PValue;
  using PState = typename RMPPolicyBase<TSpace>::PState;

  explicit RaycastingPolicy(RaycastingPolicyConfigs parameters) :
    parameters_(std::move(parameters)) {}

  virtual PValue evaluateAt(const PState& agent_state, const std::vector<PState>& obstacle_states) override;
  virtual void startEvaluateAsync(const PState& agent_state, const std::vector<PState>& obstacle_states) override {
   startEval(agent_state, obstacle_states);
   async_eval_started_ = true;
  }
  virtual void abortEvaluateAsync() override;

 private:
  void startEval(const PState& agent_state, const std::vector<PState>& obstacle_states);
  std::pair<mjtNum, Vector> raycastKernel(int ray_id,
                                          const Vector& ray_start, int target_geomtype,
                                          const mjtNum* target_pos, const mjtNum* target_rot, const mjtNum* target_size);

  const RaycastingPolicyConfigs parameters_;
  PState last_evaluated_state_;

  bool async_eval_started_ = false;
  std::vector<Matrix> metric_sum_;
  std::vector<Vector> metric_x_force_sum_;
};

}  // namespace rmpcpp

#endif  // RMPCPP_PLANNER_RAYCASTING_CUDA_H
