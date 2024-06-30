#ifndef RMPCPP_PLANNER_PLANNER_BASE_H
#define RMPCPP_PLANNER_PLANNER_BASE_H

#include "mjpc/planners/rmp/include/planner/rmp_trajectory.h"

#include <numeric>
#include <Eigen/Core>

#include "mjpc/planners/planner.h"
#include "mjpc/planners/rmp/include/core/rmp_base_policy.h"
#include "mjpc/planners/rmp/include/core/rmp_state.h"
#include "mjpc/planners/rmp/include/policies/rmp_simple_target_policy.h"

#define RMP_USE_LINEAR_GEOMETRY (1)
#define RMP_USE_SIMPLE_TARGET_POLICY (1)
#define RMP_USE_ACTUATOR_VELOCITY (1)
#define RMP_USE_ACTUATOR_MOTOR (!RMP_USE_ACTUATOR_VELOCITY)
#define RMP_BLOCKING_OBSTACLES_RATIO (0.1)
#define RMP_DISTANCE_TRACE_RAYS_NUM (100)

#define RMP_DRAW_START_GOAL (0)
#define RMP_DRAW_VELOCITY (1)
#define RMP_DRAW_TRAJECTORY (0)
#define RMP_DRAW_BLOCKING_TRACE_RAYS (0)
#define RMP_DRAW_DISTANCE_TRACE_RAYS (1)

#define RMP_KV (1)

namespace rmpcpp {

/**
 * Base class for a RMP planner
 * @tparam TSpace
 */
template <class TSpace>
class RMPPlannerBase : public mjpc::Planner {
 public:
  using VectorX = Eigen::Matrix<double, TSpace::dim, 1>;
  using VectorQ = Eigen::Matrix<double, TSpace::dim, 1>;
  using Matrix = Eigen::Matrix<double, TSpace::dim, TSpace::dim>;

  RMPPlannerBase() = default;
  virtual ~RMPPlannerBase() = default;
  virtual std::vector<std::shared_ptr<RMPPolicyBase<TSpace>>> getPolicies()
  {
      std::vector<std::shared_ptr<RMPPolicyBase<TSpace>>> policies;
      return policies;
  }

  /** Pure virtual */
  virtual void plan() = 0;

  bool success() const { return goal_reached_ && !diverged_ && !collided_; };

  virtual bool checkBlocking(const VectorQ& s1, const VectorQ& s2) = 0;
  virtual bool hasTrajectory() const = 0;
  virtual const std::shared_ptr<RMPTrajectory<TSpace>> getTrajectory() const = 0;

  virtual double distanceToObstacle(const VectorQ & pos) {
      return 0;
  }

  virtual VectorQ gradientToObstacle(const VectorQ & pos) {
      return VectorQ::Zero();
  }

 protected:
  double goal_tolerance_ = 0.005;
  bool collided_ = false;
  bool goal_reached_ = false;
  bool diverged_ = false;
};

}  // namespace rmpcpp

#endif  // RMPCPP_PLANNER_PLANNER_BASE_H
