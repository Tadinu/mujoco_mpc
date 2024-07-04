#ifndef RMPCPP_PLANNER_PLANNER_BASE_H
#define RMPCPP_PLANNER_PLANNER_BASE_H

#include <numeric>
#include <Eigen/Core>

#include "mjpc/planners/planner.h"
#include "mjpc/planners/rmp/include/core/rmp_parameters.h"
#include "mjpc/planners/rmp/include/core/rmp_state.h"
#include "mjpc/planners/rmp/include/geometry/rmp_cylindrical_geometry.h"
#include "mjpc/planners/rmp/include/geometry/rmp_linear_geometry.h"
#include "mjpc/planners/rmp/include/planner/rmp_trajectory.h"
#include "mjpc/planners/rmp/include/policies/rmp_base_policy.h"
#include "mjpc/planners/rmp/include/policies/rmp_simple_target_policy.h"

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

  using RiemannianGeometry =
#if RMP_USE_LINEAR_GEOMETRY
      LinearGeometry<TSpace::dim>;
#else
      CylindricalGeometry;
#endif
  RiemannianGeometry geometry_;
  using StateX = typename RiemannianGeometry::StateX;

  RMPPlannerBase() = default;
  virtual ~RMPPlannerBase() = default;
  virtual std::vector<std::shared_ptr<RMPPolicyBase<TSpace>>> getPolicies()
  {
      std::vector<std::shared_ptr<RMPPolicyBase<TSpace>>> policies;
      return policies;
  }

  /** Pure virtual */
  virtual void plan() = 0;

  bool success() const { return path_filled_ && !diverged_ && !collided_; };

  virtual bool checkBlocking(const VectorQ& s1, const VectorQ& s2) = 0;
  virtual bool hasTrajectory() const = 0;
  virtual std::shared_ptr<RMPTrajectory<TSpace>> getTrajectory() const = 0;

  virtual double distanceToObstacle(const VectorQ& pos) {
      return 0;
  }

  virtual VectorQ gradientToObstacle(const VectorQ& pos) {
      return VectorQ::Zero();
  }

 protected:
  double goal_tolerance_ = 0.005;
  bool collided_ = false;
  bool path_filled_ = false;
  bool diverged_ = false;
};

}  // namespace rmpcpp

#endif  // RMPCPP_PLANNER_PLANNER_BASE_H
