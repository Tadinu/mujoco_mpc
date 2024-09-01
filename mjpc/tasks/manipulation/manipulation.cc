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

#include "mjpc/tasks/manipulation/manipulation.h"

#include <absl/random/random.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>

#include <string>

#include "mjpc/planners/planner.h"
#include "mjpc/tasks/manipulation/common.h"
#include "mjpc/utilities.h"

namespace mjpc {
// task_panda_bring.xml
// task_panda_robotiq_bring.xml
std::string manipulation::Bring::XmlPath() const {
  return is_static ? GetModelPath("manipulation/task_panda_bring.xml")
                   : GetModelPath("manipulation/task_panda_dynamic.xml");
}
std::string manipulation::Bring::Name() const { return "PickAndPlace"; }

std::string manipulation::Bring::URDFPath() const {
  return mjpc::GetModelPath("manipulation/panda_with_finger.urdf");
}

void manipulation::Bring::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                               double* residual) const {
  int counter = 0;

  // reach
  double hand[3] = {0};
  ComputeRobotiqHandPos(model, data, model_vals_, hand);

  double* object = SensorByName(model, data, "object");
  mju_sub3(residual + counter, hand, object);
  counter += 3;

  // bring
  for (int i = 0; i < 8; i++) {
    double* object = SensorByName(model, data, std::to_string(i).c_str());
    double* target = SensorByName(model, data, (std::to_string(i) + "t").c_str());
    residual[counter++] = mju_dist3(object, target);
  }

  // careful
  int object_id = mj_name2id(model, mjOBJ_BODY, "object");
  residual[counter++] = CarefulCost(model, data, model_vals_, object_id);

  // away
  residual[counter++] = mju_min(0, hand[2] - 0.6);

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

void manipulation::Bring::TransitionLocked(mjModel* model, mjData* data) {
  Task::TransitionLocked(model, data);
  double residuals[100];
  double terms[10];
  residual_.Residual(model, data, residuals);
  residual_.CostTerms(terms, residuals, /*weighted=*/false);

  // bring is solved:
  if (data->time > 0 && data->userdata[0] == 0 && terms[1] < 0.04) {
    weight[0] = 0;  // disable reach
    weight[3] = 1;  // enable away

    data->userdata[0] = 1;
  }

  // away is solved, reset:
  if (data->userdata[0] == 1 && terms[3] < 0.01) {
    weight[0] = 1;  // enable reach
    weight[3] = 0;  // disable away

    absl::BitGen gen_;

    // initialise target:
    data->qpos[7 + 0] = 0.45;
    data->qpos[7 + 1] = 0;
    data->qpos[7 + 2] = 0.15;
    data->qpos[7 + 3] = absl::Uniform<double>(gen_, -1, 1);
    data->qpos[7 + 4] = absl::Uniform<double>(gen_, -1, 1);
    data->qpos[7 + 5] = absl::Uniform<double>(gen_, -1, 1);
    data->qpos[7 + 6] = absl::Uniform<double>(gen_, -1, 1);
    mju_normalize4(data->qpos + 13);

    // return stage: bring
    data->userdata[0] = 0;
  }

  // Move goals, obstacles
  if (!is_static) {
    static constexpr float amplitude = 0.5;
    double goal_curve_pos[3] = {amplitude * mju_sin(data->time), amplitude * mju_cos(data->time / mjPI), 0.3};
    SetGoalPos(goal_curve_pos);
    MoveObstacles();
  }
}

void manipulation::Bring::MoveObstacles() {
#if 0
  static constexpr bool ALTERNATING = 0;
  static int8_t sign = 1;
  if constexpr (ALTERNATING) {
    static bool sign_switched = true;
    if (fmod(data_->time, 2 * M_PI) < 0.1) {
      if (!sign_switched) {
        sign = -sign;
        sign_switched = true;
      }
    } else {
      sign_switched = false;
    }
  }
  for (auto i = 1; i <= GetTotalObstaclesNum(); ++i) {
    const auto angular_vel_i = (i > 3) ? (0.1 * i) : M_PI;
    const auto phase_i = angular_vel_i * data_->time;
    if constexpr (ALTERNATING) {
    } else {
      sign = (i % 2) == 0 ? 1 : -1;
    }
    const auto sin_pos_i = sign * mju_sin(phase_i);
    const auto cos_pos_i = mju_cos(phase_i);
    double obstacle_curve_pos[2] = {0.05 * log(i + 1) * sin_pos_i, 0.05 * log(i + 1) * cos_pos_i};
    std::ostringstream obstacle_name;
    obstacle_name << "obstacle_" << (i - 1);
    SetBodyMocapPos(obstacle_name.str().c_str(), obstacle_curve_pos);
  }
#endif
}

void manipulation::Bring::ResetLocked(const mjModel* model) {
  residual_.model_vals_ = ModelValues::FromModel(model);
}

FabPlannerConfigPtr manipulation::Bring::GetFabricsConfig() const {
  static auto config = std::make_shared<FabPlannerConfig>(FabPlannerConfig{
      .collision_geometry =
          [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
            return FabConfigExprMeta{(-4.5 / x) * (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2)};
          },
      .geometry_plane_constraint =
          [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
            return FabConfigExprMeta{(-10.0 / x) * (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2)};
          }});
  return config;
}
}  // namespace mjpc
