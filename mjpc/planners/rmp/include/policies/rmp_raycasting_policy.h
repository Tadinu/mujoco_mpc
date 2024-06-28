
#ifndef RMPCPP_PLANNER_RAYCASTING_CUDA_H
#define RMPCPP_PLANNER_RAYCASTING_CUDA_H

#include "mujoco/mujoco.h"
#include "mjpc/planners/rmp/include/core/rmp_base_policy.h"
#include "mjpc/planners/rmp/include/planner/rmp_parameters.h"

#define OUTPUT_RAYS (0)

namespace rmpcpp {

/*
 * Implements a nvblox- map based raycasting
 * obstacle avoidance policy
 */
template <class TSpace>
class RaycastingCudaPolicy : public RMPPolicyBase<TSpace> {
 public:
  using Vector = typename RMPPolicyBase<TSpace>::Vector;
  using Matrix = typename RMPPolicyBase<TSpace>::Matrix;
  using PValue = typename RMPPolicyBase<TSpace>::PValue;
  using PState = typename RMPPolicyBase<TSpace>::PState;

  RaycastingCudaPolicy(WorldPolicyParameters* parameters) :
    parameters_(*dynamic_cast<RaycastingCudaPolicyParameters*>(parameters)) {}

  virtual PValue evaluateAt(const PState& agent_state, const std::vector<PState>& obstacle_states) override;
  virtual void startEvaluateAsync(const PState& agent_state, const std::vector<PState>& obstacle_states) override {
   ispcStartEval(agent_state, obstacle_states);
   async_eval_started_ = true;
  }
  virtual void abortEvaluateAsync() override;

 private:
  void ispcStartEval(const PState& agent_state, const std::vector<PState>& obstacle_states);
  std::pair<Matrix, Vector> raycastKernel(int ray_id,
                                          const Vector& ray_start, const Vector& ray_vel, int target_geomtype,
                                          const mjtNum* target_pos, const mjtNum* target_rot, const mjtNum* target_size);

  const RaycastingCudaPolicyParameters parameters_;
  PState last_evaluated_state_;

  bool async_eval_started_ = false;
  std::vector<Matrix> metric_sum_;
  std::vector<Vector> metric_x_force_sum_;
};

}  // namespace rmpcpp

#endif  // RMPCPP_PLANNER_RAYCASTING_CUDA_H
