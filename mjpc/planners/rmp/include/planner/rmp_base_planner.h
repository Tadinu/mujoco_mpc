#ifndef RMPCPP_PLANNER_PLANNER_BASE_H
#define RMPCPP_PLANNER_PLANNER_BASE_H

#include "mjpc/planners/rmp/include/planner/rmp_trajectory.h"

#include <numeric>
#include <Eigen/Core>

#include "mjpc/planners/planner.h"
#include "mjpc/planners/rmp/include/core/rmp_base_policy.h"
#include "mjpc/planners/rmp/include/core/rmp_state.h"
#include "mjpc/planners/rmp/include/policies/rmp_simple_target_policy.h"

namespace rmpcpp {

/**
 * Base class for a RMP planner
 * @tparam TSpace
 */
template <class TSpace>
class RMPPlannerBase : public mjpc::Planner {
 public:
  using Vector = Eigen::Matrix<double, TSpace::dim, 1>;

  RMPPlannerBase() = default;
  virtual ~RMPPlannerBase() = default;
  virtual std::vector<std::shared_ptr<RMPPolicyBase<TSpace>>> getPolicies()
  {
      std::vector<std::shared_ptr<RMPPolicyBase<TSpace>>> policies;
      return policies;
  }

  /** Pure virtual */
  virtual void plan(const State<TSpace::dim>& start,
                    const Vector& end) = 0;

  Vector getGoalPos() const
  {
    return goal;
  }

  void setGoalPos(const Vector& new_goal) {
    goal = new_goal;
  }

  Vector getStartPos() const
  {
    return start;
  }

  void setStartPos(const Vector& new_start) {
    start = new_start;
  }

  Vector getStartVel() const
  {
    return start_vel;
  }

  void setStartVel(const Vector& new_start_vel) {
    start_vel = new_start_vel;
  }

  bool success() const { return goal_reached_ && !diverged_ && !collided_; };

  virtual bool checkMotion(const Vector& s1, const Vector& s2) const = 0;
  virtual bool hasTrajectory() const = 0;
  virtual const std::shared_ptr<TrajectoryRMP<TSpace>> getTrajectory() const = 0;

  virtual bool collision(const Vector& pos) {
      return false;
  }
  virtual double distanceToObstacle(const Vector& pos) {
      return 0;
  }

  virtual Vector gradientToObstacle(const Vector& pos) {
      return Vector::Zero();
  }

 protected:
  double goal_tolerance_ = 0.05;
  bool collided_ = false;
  bool goal_reached_ = false;
  bool diverged_ = false;
  Vector goal = Vector::Zero();
  Vector start = Vector::Zero();
  Vector start_vel = Vector::Zero();
};

}  // namespace rmpcpp

#endif  // RMPCPP_PLANNER_PLANNER_BASE_H
