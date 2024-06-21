
#include "mjpc/planners/rmp/include/planner/rmp_planner.h"

#include "mjpc/planners/rmp/include/core/rmp_base_policy.h"
#include "mjpc/planners/rmp/include/eval/rmp_trapezoidal_integrator.h"
#include "mjpc/planners/rmp/include/geometry/rmp_linear_geometry.h"
#include "mjpc/planners/rmp/include/planner/rmp_trajectory.h"
#include "mjpc/utilities.h"

/**
 * @tparam TSpace
 */
template <class TSpace>
void rmpcpp::RMPPlanner<TSpace>::integrate() {
  if (!trajectory_) {
    return;
  }  // ignore if trajectory is not initalized.

  LinearGeometry<TSpace::dim> geometry;
  TrapezoidalIntegrator<RMPPolicyBase<TSpace>, LinearGeometry<TSpace::dim>> integrator;

  // start from end of current trajectory (which should always be initialized
  // when this function is called)
  integrator.resetTo(trajectory_->current().position,
                     trajectory_->current().velocity);

  // reset state
  size_t num_steps = 0;

  while (!this->collided_ && !this->goal_reached_ && !this->diverged_) {
    // evaluate policies
    auto policies = this->getPolicies();
    /** Convert shared pointers to normal pointers for integration step */
    std::vector<RMPPolicyBase<TSpace>*> policiesRaw;
    policiesRaw.reserve(policies.size());
    std::transform(policies.cbegin(), policies.cend(),
                   std::back_inserter(policiesRaw),
                   [](auto &ptr) { return ptr.get(); });

    // integrate, performing over geometry on task space and pullback to config space here-in
    integrator.forwardIntegrate(policiesRaw, geometry, parameters_.dt);

    // get new positions
    Vector position, velocity, acceleration;
    integrator.getState(position, velocity, acceleration);

    // update exit conditions
    /** Collision check */
    if (!this->checkMotion(trajectory_->current().position,
                           position)) {
      this->collided_ = true;
    }

    if ((position - this->getGoalPos()).norm() <
        this->goal_tolerance_) {
      this->goal_reached_ = true;
    }

    if (num_steps > (size_t)parameters_.max_length) {
      this->diverged_ = true;
    }

    num_steps++;
    // store results
    trajectory_->addPoint(position, velocity, acceleration);
  }
}

/**
 * Start planning run
 * @tparam TSpace
 * @param start
 */
template <class TSpace>
void rmpcpp::RMPPlanner<TSpace>::plan(const State<TSpace::dim> &start,
                                      const Vector &goal) {
  // Reset states
  this->collided_ = false;
  this->goal_reached_ = false;
  this->diverged_ = false;
  trajectory_->clearData();
  trajectory_->addPoint(start.pos_, start.vel_);

  // as policies live in the world, we have to set the goal there
  this->setGoalPos(goal);

  // run integrator
  this->integrate();
}

template <class TSpace>
void rmpcpp::RMPPlanner<TSpace>::Traces(mjvScene* scn) {
  const auto current = trajectory_->current().position;
  const auto start = trajectory_->start().position;
  // make geometry
  mjpc::AddConnector(scn, mjGEOM_LINE, 5, task_->GetStartPos(),
                     task_->GetGoalPos(), (float[]){0.0, 1.0, 0.0, 1.0});

  mjpc::AddConnector(scn, mjGEOM_LINE, 5,
                   (mjtNum []){start[0], start[1], 0.01},
                   (mjtNum []){current[0], current[1], 0.01}, (float[]){1.0, 0.0, 0.0, 1.0});
}
