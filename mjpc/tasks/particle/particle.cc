// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/particle/particle.h"

#include <string>
#include <vector>

// ISPC
#include "rt_ispc.h"

// MUJOCP
#include "mujoco/mujoco.h"

// MJPC
#include "mjpc/planners/rmp/include/planner/rmp_base_planner.h"
#include "mjpc/planners/rmp/include/util/rmp_util.h"

namespace mjpc {

std::string Particle::XmlPath() const {
  return GetModelPath("particle/task_timevarying.xml");
}
std::string Particle::Name() const { return "Particle"; }

bool Particle::CheckBlocking(const double start[], const double end[]) {
#if 1
  // CHECK OVERLAPPING
  std::vector<StateX> obstacle_statesX = GetObstacleStatesX();
  double ray[3]; mju_sub3(ray, end, start);
  double ray_length = mju_normalize3(ray);
  double obstacle_i_size[3];
  int block_obstacles_count = 0;
  std::array<bool, OBSTACLES_NUM> obstacle_hits = {false};
#pragma omp parallel for if MJPC_OPENMP_ENABLED
  for (auto i = 0; i < OBSTACLES_NUM; ++i) {
    std::ostringstream obstacle_name;
    obstacle_name << "obstacle_" << i;
    //auto obstacle_i_id = mj_name2id(model_, mjOBJ_BODY, obstacle_name.str().c_str());
    auto obstacle_geom_i_id = mj_name2id(model_, mjOBJ_GEOM, obstacle_name.str().c_str());
    // Scale up obstacles size to make a contour around them by a larger margin, so safer
    mju_scl3(obstacle_i_size, &model_->geom_size[3*obstacle_geom_i_id], RMP_BLOCKING_OBSTACLES_SIZE_SCALE);

    static const double particle_size = [this]() {
      auto particle_geom_id = GetTargetObjectGeomId();
      return model_->geom_size[3*particle_geom_id];
    }();

    // add if ray-zone intersection (always true when con->pos inside zone)
#if RMP_ISPC
    const auto distance_i = ispc::raySphere(obstacle_statesX[i].pos_.data(),
                                            obstacle_i_size[0] * obstacle_i_size[0],
                                            start, ray);
#else
    const auto distance_i = mju_rayGeom(obstacle_statesX[i].pos_.data(),
                                        nullptr,
                                        obstacle_i_size,
                                        start, ray,
                                        mjGEOM_SPHERE);
#endif
    if ((distance_i != -1) && (distance_i <= ray_length)) {
#if RMP_DRAW_BLOCKING_TRACE_RAYS
      mju_copy3(ray_starts.data() + 3 * i, start);
      mju_scl3(ray, ray, ray_length);
      mju_add3(ray_ends.data() + 3 * i, start, ray);
#endif
      static constexpr float YELLOW[] = {1.0, 1.0, 0.0, 1.0};
      SetGeomColor(obstacle_geom_i_id, YELLOW);
      obstacle_hits[i] = true;
#if !MJPC_OPENMP_ENABLED
      if (++block_obstacles_count/float(OBSTACLES_NUM) >= RMP_BLOCKING_OBSTACLES_RATIO) {
        return true;
      }
#endif
    }
  }
#if MJPC_OPENMP_ENABLED
  return std::any_of(obstacle_hits.begin(), obstacle_hits.end(), [](const bool hit) {return hit;});
#else
  return false;
#endif

#else
  // CHECK COLLISION
  static int rigidmass_id = GetTargetObjectId();
  static auto obstacles_id = [this] () {
    std::array<int, OBSTACLES_NUM> obstacles_id = {0};
    for (auto i = 0; i < obstacles_id.size(); ++i) {
      std::ostringstream obstacle_name;
      obstacle_name << "obstacle_" << i;
      obstacles_id[i] = mj_name2id(model_, mjOBJ_BODY, obstacle_name.str().c_str());
    }
    return obstacles_id;
  }();
  static auto is_obstacle = [](int obj_id) {
    return std::find(std::begin(obstacles_id), std::end(obstacles_id), obj_id) != std::end(obstacles_id);
  };

  // loop over contacts
  int ncon = data_->ncon;
  std::cout << "COLLISION NUM:" << ncon << std::endl;
  for (int i = 0; i < ncon; ++i) {
    std::cout << "COLLISION NUM:" << i << ncon << std::endl;
    const mjContact* con = data_->contact + i;
    int bb[2] = {model_->geom_bodyid[con->geom[0]],
                 model_->geom_bodyid[con->geom[1]]};
    std::cout << "bb[2]:" << bb[0] << bb[1] << std::endl;
    for (int j = 0; j < 2; ++j) {
      if (is_obstacle(con->geom[j])
          && (bb[1-j] == rigidmass_id)) {
          std::cout << "COLLIDING:" << con->geom[j] << std::endl;
          return true;
        }
    }
  }
#endif
  return false;
}

// -------- Residuals for particle task -------
//   Number of residuals: 3
//     Residual (0): position - goal_position
//     Residual (1): velocity
//     Residual (2): control
// --------------------------------------------
namespace {
void ResidualImpl(const mjModel* model, const mjData* data,
                  const double goal[2], double* residual) {
  // ----- residual (0) ----- //
  double* position = SensorByName(model, data, "position");
  mju_sub(residual, position, goal, model->nq);

  // ----- residual (1) ----- //
  double* velocity = SensorByName(model, data, "velocity");
  mju_copy(residual + 2, velocity, model->nv);

  // ----- residual (2) ----- //
  mju_copy(residual + 4, data->ctrl, model->nu);
}
}  // namespace

bool Particle::QueryGoalReached()
{
  return (rmp::vectorFromScalarArray<3>(GetParticlePos()) - rmp::vectorFromScalarArray<3>(GetGoalPos())).norm() < PARTICLE_GOAL_REACH_THRESHOLD;
}

void Particle::QueryObstacleStatesX()
{
  std::lock_guard<std::mutex> lock(task_data_mutex_);
  obstacle_statesX_.clear();
  for (auto i = 0; i < OBSTACLES_NUM; ++i) {
    std::ostringstream obstacle_name;
    obstacle_name << "obstacle_" << i;
    const std::string obs_name = obstacle_name.str();
    auto obstacle_i_id = mj_name2id(model_, mjOBJ_BODY, obs_name.c_str());
    auto obstacle_geom_i_id = mj_name2id(model_, mjOBJ_GEOM, obs_name.c_str());
    mjtNum* obstacle_mocap_i_pos = QueryBodyMocapPos(model_, data_, obs_name.c_str());

    mjtNum* obstacle_i_size =  &model_->geom_size[3 * obstacle_geom_i_id];
    //mju_scl(obstacle_size, obstacle_size, 2.0, 3);
    mjtNum* obstacle_i_pos = &data_->xpos[3*obstacle_i_id];
    //mjtNum* obstacle_i_geom_pos = &data_->geom_xpos[3*obstacle_geom_i_id];
    //std::cout << obstacle_i_id << " " << obstacle_geom_i_id << std::endl;
    //std::cout<< "pos " << obs_name << ": " << obstacle_i_pos[0] << " - " << obstacle_i_pos[1] << " - " << obstacle_i_pos[2] << std::endl;
    //std::cout<< "geom_pos " << obs_name << ": " << obstacle_i_geom_pos[0] << " - " << obstacle_i_geom_pos[1] << " - " << obstacle_i_geom_pos[2] << std::endl;
    //std::cout<< "mocap_pos " << obs_name << ": " << obstacle_mocap_i_pos[0] << " - " << obstacle_mocap_i_pos[1] << " - " << obstacle_mocap_i_pos[2] << std::endl;
    mjtNum* obstacle_i_rot_mat = &data_->geom_xmat[9*obstacle_geom_i_id];
    mjtNum* obstacle_i_rot = &data_->xquat[4*obstacle_i_id];

    static constexpr int LIN_IDX = 3;
#if 0
    mjtNum obstacle_i_full_vel[6]; // rot+lin
    mj_objectVelocity(model_, data_, mjOBJ_BODY, obstacle_i_id, obstacle_i_full_vel,
                      /*flg_local=*/0);
    mjtNum obstacle_i_lin_vel[StateX::dim];
    memcpy(obstacle_i_lin_vel, &obstacle_i_full_vel[LIN_IDX], sizeof(mjtNum) * StateX::dim);
#else
    mjtNum obstacle_i_lin_vel[StateX::dim];
    mju_copy(obstacle_i_lin_vel, &data_->cvel[6*obstacle_i_id + LIN_IDX], StateX::dim);
#endif
    obstacle_statesX_.push_back(StateX{.pos_ = rmp::vectorFromScalarArray<StateX::dim>(obstacle_i_pos),
                                       .rot_ = rmp::quatFromScalarArray<StateX::dim>(obstacle_i_rot).toRotationMatrix(),
                                       .vel_ = rmp::vectorFromScalarArray<StateX::dim>(obstacle_i_lin_vel),
                                       .size_ = rmp::vectorFromScalarArray<StateX::dim>(obstacle_i_size)});
  }
}

void Particle::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  // some Lissajous curve
  double goal[2]{0.25 * mju_sin(data->time), 0.25 * mju_cos(data->time / mjPI)};
  ResidualImpl(model, data, goal, residual);
}

void Particle::TransitionLocked(mjModel* model, mjData* data) {
  model_ = model;
  data_ = data;
  QueryObstacleStatesX();
  last_goal_reached_ = QueryGoalReached();
  // some Lissajous curve
  double goal_curve_pos[2] = {0.25 * mju_sin(data->time),
                              0.25 * mju_cos(data->time / mjPI)};

  // update mocap position
  SetGoalPos(goal_curve_pos);
  MoveObstacles();
}

void Particle::ModifyScene(const mjModel* model, const mjData* data,
                           mjvScene* scene) const {
#if RMP_DRAW_START_GOAL
  static constexpr float GREEN[] = {0.0, 1.0, 0.0, 1.0};
  mjpc::AddConnector(scene, mjGEOM_LINE, 5, GetStartPos(),
                     GetGoalPos(), GREEN);
#endif

#if RMP_DRAW_BLOCKING_TRACE_RAYS
  if (!ray_starts.empty() && !ray_ends.empty()) {
    static constexpr float PINK[] = {1.0, 0.5, 1.0, 0.5};
    for (auto i = 0; i < OBSTACLES_NUM; ++i) {
      mjpc::AddConnector(scene, mjGEOM_LINE, 3,
                         ray_starts.data() + 3 * i,
                         ray_ends.data() + 3 * i, PINK);
    }
    const_cast<Particle*>(this)->ray_starts.fill({0.});
    const_cast<Particle*>(this)->ray_ends.fill({0.});
  }
#endif
}

std::string ParticleFixed::XmlPath() const {
  return GetModelPath("particle/task_timevarying.xml");
}
std::string ParticleFixed::Name() const { return "ParticleFixed"; }

void ParticleFixed::TransitionLocked(mjModel* model, mjData* data) {
  model_ = model;
  data_ = data;
  QueryObstacleStatesX();
  last_goal_reached_ = QueryGoalReached();
  if (this->last_goal_reached_) {
    // Stop the particle
    data->ctrl[0] = 0.f;
    data->ctrl[1] = 0.f;
    // Randomize goal and obstacles for the next run
    double new_goal_pos[2] = {rand_val(), rand_val()};
    SetGoalPos(new_goal_pos);
    RandomizeObstacles();
  }
}

void ParticleFixed::FixedResidualFn::Residual(const mjModel* model,
                                              const mjData* data,
                                              double* residual) const {
  ResidualImpl(model, data, particle_fixed_task->QueryGoalPos(), residual);
}
}  // namespace mjpc
