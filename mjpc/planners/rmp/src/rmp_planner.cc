
#include "mjpc/planners/rmp/include/planner/rmp_planner.h"

/**
 * @tparam TSpace
 */
template <class TSpace>
void rmpcpp::RMPPlanner<TSpace>::integrate() {
  // ignore if trajectory is not initialized
  if (!model_ || !data_ || !trajectory_) {
    return;
  }

  std::vector<StateX> obstacle_statesX = task_->GetObstacleStatesX();

  // start from end of current trajectory (which should always be initialized
  // when this function is called)
  integrator_.resetTo(trajectory_->hasData() ? trajectory_->current().position : GetStartPosQ(),
                      trajectory_->hasData() ? trajectory_->current().velocity : GetStartVelQ(),
                      std::move(obstacle_statesX));

  // reset state
  size_t num_steps = 0;

  // Build a list of waypoints leading to goal without collision or divergence
  while (!this->collided_ && !this->path_filled_ && !this->diverged_) {
    // Evaluate policies
    auto policies = this->getPolicies();
    // Convert shared pointers to normal pointers for integration step
    std::vector<RMPPolicyBase<TSpace>*> policiesRaw;
    policiesRaw.reserve(policies.size());
    std::transform(policies.cbegin(), policies.cend(),
                   std::back_inserter(policiesRaw),
                   [](auto &ptr) { return ptr.get(); });

    // Integrate, performing over geometry on task space and pullback to config space here-in
    integrator_.forwardIntegrate(policiesRaw, this->geometry_, parameters_.dt);

    // Get next state
    VectorQ next_position, next_velocity, next_acceleration;
    integrator_.getState(next_position, next_velocity, next_acceleration);

    // Update exit conditions
    // Meant to be moved so not defined const here
    RMPWaypoint<TSpace> next_waypoint = {.position = next_position,
                                         .velocity = next_velocity,
                                         .acceleration = next_acceleration};
    // Collision check
    if (trajectory_->hasData() && this->checkBlocking(trajectory_->current().position, next_position)) {
      this->collided_ = true;
    }

    if ((next_position - this->GetGoalPosQ()).norm() < this->goal_tolerance_) {
      this->path_filled_ = true;
    }

    if ((num_steps++ > size_t(parameters_.max_steps)) &&
        (trajectory_->current().cumulative_length > double(parameters_.max_length))) {
      this->diverged_ = true;
    }

    // Update [trajectory_]
    trajectory_->addPoint(std::move(next_waypoint));
  }
  trajectory_->setCollided(this->collided_);
}

/**
 * Start planning run
 * @tparam TSpace
 * @param start
 */
template <class TSpace>
void rmpcpp::RMPPlanner<TSpace>::plan() {
  // Reset states
  this->collided_ = false;
  this->path_filled_ = false;
  this->diverged_ = false;

  trajectory_->clearData();
  trajectory_->addPoint(GetStartPosQ(), GetStartVelQ());

  // run integrator
  this->integrate();
}

template<class TSpace>
void rmpcpp::RMPPlanner<TSpace>::DrawTrajectoryCurrent(const mjpc::Trajectory* trajectory,
                                                       mjvScene* scn, const float* color)
{
  const auto* rmp_trajectory = static_cast<const RMPTrajectory<TSpace>*>(trajectory);
  auto currentPoint = rmp_trajectory->current();
  static constexpr float ORANGE[] = {1.0, 165.0/255.0, 0.0, 1.0};
  if (!color) {
    color = ORANGE;
  }
  const auto curPos = this->geometry_.convertPosToX(currentPoint.position);
  const auto curVel = this->geometry_.convertPosToX(currentPoint.velocity);
  const VectorX curVelEnd = curPos + curVel.norm() * curVel;
  mjpc::AddConnector(scn ? scn : task_->scene_, mjGEOM_ARROW, 0.005, curPos.data(),
                     curVelEnd.data(), color);
}

template<class TSpace>
void rmpcpp::RMPPlanner<TSpace>::DrawTrajectory(const mjpc::Trajectory* trajectory,
                                                mjvScene* scn, const float* color)
{
  const auto* rmp_trajectory = static_cast<const RMPTrajectory<TSpace>*>(trajectory);
  static constexpr float RED[] = {1.0, 0.0, 0.0, 1.0};
  if (!color) {
    color = RED;
  }
  auto scene = scn ? scn : task_->scene_;
  for(auto i = 0; i < rmp_trajectory->getWaypointsCount() - 1; ++i) {
    const auto point_i = this->geometry_.convertPosToX((*rmp_trajectory)[i].position);
    const auto point_i_1 = this->geometry_.convertPosToX((*rmp_trajectory)[i+1].position);
    mjpc::AddConnector(scene, mjGEOM_LINE, 5,
                       point_i.data(),
                       point_i_1.data(), color);
  }
}

template <class TSpace>
void rmpcpp::RMPPlanner<TSpace>::Traces(mjvScene* scn) {
  //static constexpr float PINK[] = {1.0, 192.0/255.0,203.0/255.0, 1.0}
#if RMP_DRAW_VELOCITY
  DrawTrajectoryCurrent(BestTrajectory(), scn);
#endif

#if RMP_DRAW_DISTANCE_TRACE_RAYS
  //static constexpr float PINK[] = {1.0, 0.5, 1.0, 0.5};
  for (auto& policy : getPolicies()) {
    for (const auto& trace : policy->raytraces_) {
      mjpc::AddConnector(scn, mjGEOM_LINE, 0.5,
                         trace.ray_start.data(),
                         trace.ray_end.data(),
                         (float[]){float(trace.distance/0.3),
                                   0.5, 0.0, 0.5});
    }
    policy->raytraces_.clear();
  }
#endif

#if RMP_DRAW_TRAJECTORY
  DrawTrajectory(trajectory_.get(), scn);
#endif
}
