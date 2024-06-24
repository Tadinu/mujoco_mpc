
#include "mjpc/planners/rmp/include/planner/rmp_planner.h"

/**
 * @tparam TSpace
 */
template <class TSpace>
void rmpcpp::RMPPlanner<TSpace>::integrate() {
  if (!trajectory_) {
    return;
  }  // ignore if trajectory is not initalized.

  // start from end of current trajectory (which should always be initialized
  // when this function is called)
  integrator_.resetTo(trajectory_->current().position,
                      trajectory_->current().velocity);

  // reset state
  size_t num_steps = 0;

  // Build a list of waypoints leading to goal without collision or divergence
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
    integrator_.forwardIntegrate(policiesRaw, geometry_, parameters_.dt);

    // get next state
    VectorQ next_position, next_velocity, next_acceleration;
    integrator_.getState(next_position, next_velocity, next_acceleration);

    // update exit conditions
    /** Collision check */
    if (this->checkBlocking(trajectory_->current().position, next_position)) {
      this->collided_ = true;
    }

    if ((next_position - this->getGoalPos()).norm() < this->goal_tolerance_) {
      this->goal_reached_ = true;
    }

    if ((num_steps > (size_t)parameters_.max_steps) &&
        (trajectory_->current().cumulative_length > (size_t)parameters_.max_length)) {
      this->diverged_ = true;
    }

    num_steps++;
    // store results
    trajectory_->addPoint(next_position, next_velocity, next_acceleration);
  }
}

/**
 * Start planning run
 * @tparam TSpace
 * @param start
 */
template <class TSpace>
void rmpcpp::RMPPlanner<TSpace>::plan(const State<TSpace::dim> &start) {
  // Reset states
  this->collided_ = false;
  this->goal_reached_ = false;
  this->diverged_ = false;

  trajectory_->clearData();
  trajectory_->addPoint(start.pos_, start.vel_);

  // run integrator
  this->integrate();
}

template <class TSpace>
void rmpcpp::RMPPlanner<TSpace>::Traces(mjvScene* scn) {
  //static constexpr float PINK[] = {1.0, 192.0/255.0,203.0/255.0, 1.0};

#if RMP_DRAW_VELOCITY
  const auto* best_trajectory = static_cast<const TrajectoryRMP<TSpace>*>(BestTrajectory());
  auto currentPoint = best_trajectory->current();
  static constexpr float ORANGE[] = {1.0, 165.0/255.0, 0.0, 1.0};
  //static constexpr float ORANGE[] = {0.0, 0.0, 1.0, 1.0};
  auto curPos = geometry_.convertPosToX(currentPoint.position);
  auto curVel = geometry_.convertPosToX(currentPoint.velocity);
  curVel.normalize();
  mjpc::AddConnector(task_->scn, mjGEOM_ARROW, 0.005, curPos.data(),
                     VectorX(curPos + 0.05 * curVel).data(), ORANGE);
#endif

#if RMP_DRAW_START_GOAL
  const auto start_pos = geometry_.convertPosToX(this->getStartPos());
  const auto goal_pos = geometry_.convertPosToX(this->getGoalPos());

  static constexpr float GREEN[] = {0.0, 1.0, 0.0, 1.0};
  mjpc::AddConnector(scn, mjGEOM_LINE, 5, start_pos.data(),
                     goal_pos.data(), GREEN);
#endif

#if RMP_DRAW_TRAJECTORY
  static constexpr float RED[] = {1.0, 0.0, 0.0, 1.0};
  for(auto i = 0; i < trajectory_->getWaypointsCount() - 1; ++i) {
    const auto point_i = geometry_.convertPosToX((*trajectory_)[i].position);
    const auto point_i_1 = geometry_.convertPosToX((*trajectory_)[i+1].position);
    mjpc::AddConnector(scn, mjGEOM_LINE, 5,
                       point_i.data(),
                       point_i_1.data(), RED);
  }
#endif
}
