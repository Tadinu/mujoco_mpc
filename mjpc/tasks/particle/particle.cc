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

#include <mujoco/mujoco.h>
#include "mjpc/planners/rmp/include/planner/rmp_base_planner.h"

namespace mjpc {

std::string Particle::XmlPath() const {
  return GetModelPath("particle/task_timevarying.xml");
}
std::string Particle::Name() const { return "Particle"; }

bool Particle::CheckBlocking(const double start[], const double end[]) {
#if 1
  static constexpr double TRACE_DELTA = 2*M_PI/ TRACE_RAYS_NUM;

  // CHECK OVERLAPPING
  double vec[3]; mju_sub3(vec, end, start);
  double length = mju_normalize3(vec);
  double obstacle_size[3];
  int block_obstacles_count = 0;
  bool blocked = false;
  for (auto i = 0; i < OBSTACLES_NUM; ++i) {
    blocked = false;
    std::ostringstream obstacle_name;
    obstacle_name << "obstacle_" << i;
    auto obstacle_i_id = mj_name2id(model_, mjOBJ_BODY, obstacle_name.str().c_str());
    auto obstacle_geom_i_id = mj_name2id(model_, mjOBJ_GEOM, obstacle_name.str().c_str());
    //mju_scl(obstacle_size, &model_->geom_size[3*obstacle_geom_i_id], 2.0, 3);

    static const double particle_size = [this]() {
      auto particle_geom_id = GetTargetObjectGeomId();
      return model_->geom_size[3*particle_geom_id];
    }();

    for (auto j = 0; j < TRACE_RAYS_NUM; ++j)
    {
      mjtNum pt[3];
      pt[0] = start[0] + particle_size * mju_sin(j* TRACE_DELTA);
      pt[1] = start[1] + particle_size * mju_cos(j* TRACE_DELTA);
      pt[2] = start[2];

#if RMP_DRAW_BLOCKING_TRACE_RAYS
      mju_copy3(ray_start.data() + i * TRACE_RAYS_NUM + 3 * j, pt);
      mju_add3(ray_end.data() + i * TRACE_RAYS_NUM + 3 * j, pt, vec);
#endif

      // add if ray-zone intersection (always true when con->pos inside zone)
      const auto distance = mju_rayGeom(&data_->xpos[3*obstacle_i_id], &data_->xmat[9*obstacle_i_id],
                      obstacle_size,
                      pt, vec,
                      mjGEOM_SPHERE);
      if ((distance != -1) && (distance < length)) {
        static constexpr float YELLOW[] = {1.0, 1.0, 0.0, 1.0};
        SetGeomColor(obstacle_geom_i_id, YELLOW);
        if (block_obstacles_count/float(OBSTACLES_NUM) >= RMP_BLOCKING_OBSTACLES_RATIO) {
          return true;
        }
        blocked = true;
        break;
      }
    }

    if (blocked) {
      ++block_obstacles_count;
    }
  }
  return false;
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

void Particle::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  // some Lissajous curve
  double goal[2]{0.25 * mju_sin(data->time), 0.25 * mju_cos(data->time / mjPI)};
  ResidualImpl(model, data, goal, residual);
}

void Particle::TransitionLocked(mjModel* model, mjData* data) {
  // some Lissajous curve
  double goal_curve_pos[2] = {0.25 * mju_sin(data->time),
                              0.25 * mju_cos(data->time / mjPI)};

  // update mocap position
  SetGoalPos(goal_curve_pos);
  MoveObstacles();
}

std::string ParticleFixed::XmlPath() const {
  return GetModelPath("particle/task_timevarying.xml");
}
std::string ParticleFixed::Name() const { return "ParticleFixed"; }

void ParticleFixed::FixedResidualFn::Residual(const mjModel* model,
                                              const mjData* data,
                                              double* residual) const {
  ResidualImpl(model, data, particle_fixed_task->GetGoalPos(), residual);
}
}  // namespace mjpc
