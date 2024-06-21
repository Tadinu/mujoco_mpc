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
#include "mjpc/utilities.h"

namespace mjpc {

std::string Particle::XmlPath() const {
  return GetModelPath("particle/task_timevarying.xml");
}
std::string Particle::Name() const { return "Particle"; }

bool Particle::checkCollision(double pos[]) const {
  //mjs_getDefault(mjs_findBody(model_, "pointmass")->element);
  //mjs_getDefault(mjs_findBody(model_, "obstacle_1")->element);
  static int pointmass = mj_name2id(model_, mjOBJ_BODY, "pointmass");
  static auto obstacles = [this] () {
    std::array<int, 7> obstacles = {0};
    for (auto i = 0; i < obstacles.size(); ++i) {
      std::ostringstream obstacle_name;
      obstacle_name << "obstacle_" << i;
      obstacles[i] = mj_name2id(model_, mjOBJ_BODY, obstacle_name.str().c_str());
    }
    return obstacles;
  }();
  static auto has_obstacle = [](int obstacle_id) {
      return std::find(std::begin(obstacles), std::end(obstacles), obstacle_id) != std::end(obstacles);
  };

  // loop over contacts
  int ncon = data_->ncon;
  for (int i = 0; i < ncon; ++i) {
    const mjContact* con = data_->contact + i;
    int bb[2] = {model_->geom_bodyid[con->geom[0]],
                 model_->geom_bodyid[con->geom[1]]};
    for (int j = 0; j < 2; ++j) {
      if (has_obstacle(con->geom[j])
          && (bb[1-j] == pointmass)) {
          return true;
        }
    }
  }
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
  double goal[2]{0.25 * mju_sin(data->time), 0.25 * mju_cos(data->time / mjPI)};

  // update mocap position
  data->mocap_pos[0] = goal[0];
  data->mocap_pos[1] = goal[1];
}

std::string ParticleFixed::XmlPath() const {
  return GetModelPath("particle/task_timevarying.xml");
}
std::string ParticleFixed::Name() const { return "ParticleFixed"; }

void ParticleFixed::FixedResidualFn::Residual(const mjModel* model,
                                              const mjData* data,
                                              double* residual) const {
  double goal[2]{data->mocap_pos[0], data->mocap_pos[1]};
  ResidualImpl(model, data, goal, residual);
}
}  // namespace mjpc
