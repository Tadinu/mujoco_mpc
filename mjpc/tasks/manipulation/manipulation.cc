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
std::string manipulation::Bring::XmlPath() const { return GetModelPath("manipulation/task_panda_bring.xml"); }
std::string manipulation::Bring::Name() const { return "PickAndPlace"; }

std::string manipulation::Bring::URDFPath() const {
  // "panda_with_finger.urdf"
  return mjpc::GetModelPath("manipulation/panda_for_fk.urdf");
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
  if (!IsGoalFixed()) {
    static constexpr float amplitude = 0.5;
    SetGoalPos((double[]){amplitude * mju_sin(data->time / 2), amplitude * mju_cos(data->time / mjPI), 0.7});

    mjtNum quat[4];
    mju_axisAngle2Quat(quat, (mjtNum[]){0., 1., 0.}, M_PI_2 * mju_sin(data->time / 2));
    SetGoalQuat(quat);
  }

  if (!AreObstaclesFixed()) {
    MoveObstacles();
  }
}

void manipulation::Bring::MoveObstacles() {
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
  std::vector<StateX> obstacle_statesX = GetObstacleStatesX();
  for (auto i = 1; i <= obstacle_statesX.size(); ++i) {
    const auto angular_vel_i = (i > 3) ? (0.1 * i) : M_PI;
    const auto phase_i = 0.5 * angular_vel_i * data_->time;
    if constexpr (ALTERNATING) {
    } else {
      sign = (i % 2) == 0 ? 1 : -1;
    }
    const auto sin_pos_i = sign * mju_sin(phase_i);
    const auto cos_pos_i = mju_cos(phase_i);

    double obstacle_curve_pos[3];
    if (i % 2 == 0) {
      mju_copy3(obstacle_curve_pos, (double[]){0.5 * log(i + 1) * sin_pos_i, 0.5 * log(i + 1) * cos_pos_i,
                                               0.7 + 0.5 * log(i + 1) * cos_pos_i});
    } else {
      mju_copy3(obstacle_curve_pos, (double[]){0.5 * log(i + 1) * sin_pos_i, 0.5 * log(i + 1) * cos_pos_i,
                                               0.7 + 0.5 * log(i + 1) * sin_pos_i});
    }
    std::ostringstream obstacle_name;
    obstacle_name << "obstacle_" << (i - 1);
    SetBodyMocapPos(obstacle_name.str().c_str(), obstacle_curve_pos);
  }
}

void manipulation::Bring::ResetLocked(const mjModel* model) {
  residual_.model_vals_ = ModelValues::FromModel(model);
}

FabPlannerConfigPtr manipulation::Bring::GetFabricsConfig() const {
  const FabConfigFunc fAttractor_potential = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    // alpha: scaling factor for the softmax
    static constexpr float alpha = 10.f;
    const CaSX x_norm = CaSX::norm_2(x);
    return FabConfigExprMeta{5.0 * (x_norm + (1 / alpha) * CaSX::log(1 + CaSX::exp(-2 * alpha * x_norm)))};
  };
  const FabConfigFunc fAttractor_metric = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    static constexpr float alpha = 600.f;
    static constexpr float beta = 0.3;
    const CaSX x_norm = CaSX::norm_2(x);
    return FabConfigExprMeta{((alpha - beta) * CaSX::exp(-1 * CaSX::pow(0.75 * x_norm, 2)) + beta) *
                             fab_math::CASX_IDENTITY(x.size().first)};
  };

  FabConfigFunc fdamper_beta = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    const auto a_ex = FabPlannerConfig::sym_var("a_ex", affix);
    const auto a_le = FabPlannerConfig::sym_var("a_le", affix);
    return FabConfigExprMeta{
        0.5 * (CaSX::tanh(-0.5 * (CaSX::norm_2(x) - 0.02)) + 1) * 0.065 + 0.01 + CaSX::fmax(0, a_ex - a_le),
        {a_ex.name(), a_le.name()}};
  };

  FabConfigFunc fdamper_eta = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    return FabConfigExprMeta{0.5 * (CaSX::tanh(-0.9 * 0.5 * CaSX::dot(xdot, xdot) - 0.5) + 1)};
  };

  auto config = std::make_shared<FabPlannerConfig>(FabPlannerConfig{
      /*
      .collision_geometry =
          [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
            return FabConfigExprMeta{(-4.5 / x) * (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2)};
          },
      .geometry_plane_constraint =
          [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
            return FabConfigExprMeta{(-10.0 / x) * (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2)};
          },
        */
      .attractor_potential = fAttractor_potential,
      .attractor_metric = fAttractor_metric,
      .damper_beta = fdamper_beta,
      .damper_eta = fdamper_eta});
  return config;
}
}  // namespace mjpc
