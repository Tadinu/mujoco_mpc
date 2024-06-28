
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

  std::vector<StateX> obstacle_states;
  for (auto i = 0; i < task_->OBSTACLES_NUM; ++i) {
    std::ostringstream obstacle_name;
    obstacle_name << "obstacle_" << i;
    auto obstacle_i_id =
        mj_name2id(model_, mjOBJ_BODY, obstacle_name.str().c_str());
    auto obstacle_geom_i_id =
        mj_name2id(model_, mjOBJ_GEOM, obstacle_name.str().c_str());

    mjtNum* obstacle_i_size =  &model_->geom_size[3 * obstacle_geom_i_id];
    //mju_scl(obstacle_size, obstacle_size, 2.0, 3);
    mjtNum* obstacle_i_pos = &data_->xpos[3*obstacle_i_id];
    mjtNum* obstacle_i_rot = &data_->xmat[9*obstacle_i_id];

    static constexpr int LIN_IDX = 3;
#if 0
    mjtNum obstacle_i_full_vel[6]; // rot+lin
    mj_objectVelocity(model_, data_, mjOBJ_BODY, obstacle_i_id, obstacle_i_full_vel,
                      /*flg_local=*/0);
    mjtNum obstacle_i_lin_vel[TSpace::dim];
    memcpy(obstacle_i_lin_vel, &obstacle_i_full_vel[LIN_IDX], sizeof(mjtNum) * TSpace::dim);
#else
    mjtNum obstacle_i_lin_vel[TSpace::dim];
    mju_copy(obstacle_i_lin_vel, &data_->cvel[6*obstacle_i_id + LIN_IDX], TSpace::dim);
#endif
    obstacle_states.push_back(StateX{.pos_ = vectorFromScalarArray<TSpace::dim>(obstacle_i_pos),
                                     .rot_ = matrixFromScalarArray<TSpace::dim>(obstacle_i_rot),
                                     .vel_ = vectorFromScalarArray<TSpace::dim>(obstacle_i_lin_vel),
                                     .size_ = vectorFromScalarArray<TSpace::dim>(obstacle_i_size)});
  }

  // start from end of current trajectory (which should always be initialized
  // when this function is called)
  integrator_.resetTo(trajectory_->hasData() ? trajectory_->current().position : GetStartPos(),
                      trajectory_->hasData() ? trajectory_->current().velocity : GetStartVel(),
                      std::move(obstacle_states));

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
    // Meant to be moved so not defined const here
    RMPWaypoint<TSpace> next_waypoint = {.position = next_position,
                                         .velocity = next_velocity,
                                         .acceleration = next_acceleration};
    // collision check
    if (trajectory_->hasData() && this->checkBlocking(trajectory_->current().position, next_position)) {
      this->collided_ = true;
    }

    if ((next_position - geometry_.convertPosToQ(vectorFromScalarArray<TSpace::dim>(task_->GetGoalPos()))).norm() < this->goal_tolerance_) {
      this->goal_reached_ = true;
    }

    if ((num_steps > (size_t)parameters_.max_steps) &&
        (trajectory_->current().cumulative_length > (size_t)parameters_.max_length)) {
      this->diverged_ = true;
    }

    num_steps++;
    // store results
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
  this->goal_reached_ = false;
  this->diverged_ = false;

  trajectory_->clearData();
  trajectory_->addPoint(GetStartPos(), GetStartVel());

  // run integrator
  this->integrate();
}

template<class TSpace>
void rmpcpp::RMPPlanner<TSpace>::DrawTrajectoryCurrent(const mjpc::Trajectory* trajectory, const float* color)
{
  const auto* rmp_trajectory = static_cast<const RMPTrajectory<TSpace>*>(trajectory);
  auto currentPoint = rmp_trajectory->current();
  static constexpr float ORANGE[] = {1.0, 165.0/255.0, 0.0, 1.0};
  if (!color) {
    color = ORANGE;
  }
  const auto curPos = geometry_.convertPosToX(currentPoint.position);
  const auto curVel = geometry_.convertPosToX(currentPoint.velocity);
  const VectorX curVelEnd = curPos + curVel.norm() * curVel;
  mjpc::AddConnector(task_->scene_, mjGEOM_ARROW, 0.005, curPos.data(),
                     curVelEnd.data(), color);
}

template<class TSpace>
void rmpcpp::RMPPlanner<TSpace>::DrawTrajectory(const mjpc::Trajectory* trajectory, const float* color)
{
  const auto* rmp_trajectory = static_cast<const RMPTrajectory<TSpace>*>(trajectory);
  static constexpr float RED[] = {1.0, 0.0, 0.0, 1.0};
  if (!color) {
    color = RED;
  }
  for(auto i = 0; i < rmp_trajectory->getWaypointsCount() - 1; ++i) {
    const auto point_i = geometry_.convertPosToX((*rmp_trajectory)[i].position);
    const auto point_i_1 = geometry_.convertPosToX((*rmp_trajectory)[i+1].position);
    mjpc::AddConnector(task_->scene_, mjGEOM_LINE, 5,
                       point_i.data(),
                       point_i_1.data(), color);
  }
}

template <class TSpace>
void rmpcpp::RMPPlanner<TSpace>::Traces(mjvScene* scn) {
  //static constexpr float PINK[] = {1.0, 192.0/255.0,203.0/255.0, 1.0}
#if RMP_DRAW_VELOCITY
  DrawTrajectoryCurrent(BestTrajectory());
#endif

#if RMP_DRAW_DISTANCE_TRACE_RAYS
  //static constexpr float PINK[] = {1.0, 0.5, 1.0, 0.5};
  for (auto& policy : getPolicies()) {
    for (const auto& trace : policy->raytraces_) {
      mjpc::AddConnector(scn, mjGEOM_LINE, 0.5,
                         trace.ray_start.data(),
                         trace.ray_end.data(), (float[]){float(mju_sigmoid(trace.distance)), 0.0, 0.0, 0.5});
    }
    policy->raytraces_.clear();
  }
#endif

#if RMP_DRAW_BLOCKING_TRACE_RAYS
  static const int OBSTACLES_NUM = task_->OBSTACLES_NUM;
  static const int PTS_NUM = task_->TRACE_RAYS_NUM;

  if (task_->ray_start.size() && task_->ray_end.size()) {
    static constexpr float PINK[] = {1.0, 0.5, 1.0, 0.5};
    for (auto i = 0; i < OBSTACLES_NUM; ++i) {
      for (auto j = 0; j < PTS_NUM; ++j) {
        mjpc::AddConnector(scn, mjGEOM_LINE, 0.3,
                           task_->ray_start.data() + i * PTS_NUM + 3 * j,
                           task_->ray_end.data() + i * PTS_NUM + 3 * j, PINK);
      }
    }
    task_->ray_start.fill(0.);
    task_->ray_end.fill(0.);
  }
#endif

#if RMP_DRAW_TRAJECTORY
  DrawTrajectory(trajectory_.get());
#endif

#if RMP_DRAW_START_GOAL
  const auto start_pos = task_->GetStartPos();
  const auto goal_pos = task_->GetGoalPos();

  static constexpr float GREEN[] = {0.0, 1.0, 0.0, 1.0};
  mjpc::AddConnector(scene_, mjGEOM_LINE, 5, start_pos.data(),
                     goal_pos.data(), GREEN);
#endif
}
