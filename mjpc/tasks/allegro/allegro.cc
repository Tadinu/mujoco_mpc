// Copyright 2024 DeepMind Technologies Limited
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

#include "mjpc/tasks/allegro/allegro.h"

#include <string>

// mujoco
#include <mujoco/mujoco.h>

// drake
#include <drake/geometry/proximity_properties.h>
#include <drake/multibody/parsing/parser.h>

// mjpc
#include "mjpc/utilities.h"

namespace mjpc {
std::string Allegro::XmlPath() const {
  return GetModelPath("allegro/task.xml");
}

std::string Allegro::Name() const { return "Allegro"; }

// ------- Residuals for target manipulation task ------
//     Cube position: (3)
//     Cube orientation: (3)
//     Cube linear velocity: (3)
//     Control: (16), there are 16 servos
//     Nominal pose: (16)
//     Joint velocity: (16)
// ------------------------------------------
void Allegro::ResidualFn::Residual(const mjModel* model, const mjData* data, double* residual) const {
  int counter = 0;
  const auto target_prefix = dynamic_cast<const Allegro*>(task_)->target_type_name();

  // ---------- Cube position ----------
  double* target_position = SensorByName(model, data, target_prefix + "_position");
  double* target_goal_position = SensorByName(model, data, target_prefix + "_goal_position");

  mju_sub3(residual + counter, target_position, target_goal_position);
  counter += 3;

  // ---------- Cube orientation ----------
  double* target_orientation = SensorByName(model, data, target_prefix + "_orientation");
  double* goal_target_orientation = SensorByName(model, data, target_prefix + "_goal_orientation");
  mju_normalize4(goal_target_orientation);

  mju_subQuat(residual + counter, goal_target_orientation, target_orientation);
  counter += 3;

  // ---------- Cube linear velocity ----------
  double* target_linear_velocity = SensorByName(model, data, target_prefix + "_linear_velocity");

  mju_copy(residual + counter, target_linear_velocity, 3);
  counter += 3;

  // ---------- Control ----------
  mju_copy(residual + counter, data->actuator_force, model->nu);
  counter += model->nu;

  // ---------- Nominal Pose ----------
  mju_sub(residual + counter, data->qpos + 7, model->key_qpos + 7, 16);
  counter += 16;

  // ---------- Joint Velocity ----------
  mju_copy(residual + counter, data->qvel + 6, 16);
  counter += 16;

  // Sanity check
  CheckSensorDim(model, counter);
}

void Allegro::TransitionLocked(mjModel* model, mjData* data) {
  // Check for contact between the target and the floor
  int target_geom = mj_name2id(model, mjOBJ_GEOM, target_geom_name().c_str());
  int floor = mj_name2id(model, mjOBJ_GEOM, "floor");

  bool on_floor = false;
  for (int i = 0; i < data->ncon; i++) {
    mjContact* g = data->contact + i;
    if ((g->geom1 == target_geom && g->geom2 == floor) || (g->geom2 == target_geom && g->geom1 == floor)) {
      on_floor = true;
      break;
    }
  }

  // If the target is on the floor and not moving, reset it
  double* target_lin_vel = SensorByName(model, data, target_type_name() + "_linear_velocity");
  if (on_floor && (mju_norm3(target_lin_vel) < 0.001)) {
    int target_body = mj_name2id(model, mjOBJ_BODY, target_body_name().c_str());
    if (target_body != -1) {
      int jnt_qposadr = model->jnt_qposadr[model->body_jntadr[target_body]];
      int jnt_veladr = model->jnt_dofadr[model->body_jntadr[target_body]];
      mju_copy(data->qpos + jnt_qposadr, model->qpos0 + jnt_qposadr, 7);
      mju_zero(data->qvel + jnt_veladr, 6);
    }

    // Step the simulation forward
    mutex_.unlock();
    mj_forward(model, data);
    mutex_.lock();
  }
}
} // namespace mjpc
