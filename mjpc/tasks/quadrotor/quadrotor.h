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

#ifndef MJPC_TASKS_QUADROTOR_QUADROTOR_H_
#define MJPC_TASKS_QUADROTOR_QUADROTOR_H_

#include <memory>
#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
class Quadrotor : public Task {
 public:
  Quadrotor() : residual_(this) {}
  inline std::string XmlPath() const override {
    return GetModelPath("quadrotor/task.xml");
  }
  inline std::string Name() const override { return "Quadrotor"; }

  class ResidualFn : public BaseResidualFn {
   public:
    ResidualFn(const ResidualFn& residual) = default;
    explicit ResidualFn(const Quadrotor* task) : BaseResidualFn(task) {}
    // --------------- Residuals for quadrotor task ---------------
    //   Number of residuals: 5
    //     Residual (0): position - goal position
    //     Residual (1): orientation - goal orientation
    //     Residual (2): linear velocity - goal linear velocity
    //     Residual (3): angular velocity - goal angular velocity
    //     Residual (4): control
    //   Number of parameters: 6
    // ------------------------------------------------------------
    inline void Residual(const mjModel* model, const mjData* data,
                         double* residuals) const override {
      // ---------- Residual (0) ----------
      double* position = SensorByName(model, data, "position");
      mju_sub(residuals, position, data->mocap_pos, 3);

      // ---------- Residual (1) ----------
      double* linear_velocity = SensorByName(model, data, "linear_velocity");
      mju_copy(residuals + 3, linear_velocity, 3);

      // ---------- Residual (2) ----------
      double* angular_velocity = SensorByName(model, data, "angular_velocity");
      mju_copy(residuals + 6, angular_velocity, 3);

      // ---------- Residual (3) ----------
      double thrust = (model->body_mass[0] + model->body_mass[1]) *
                      mju_norm3(model->opt.gravity) / model->nu;
      for (int i = 0; i < model->nu; i++) {
        residuals[9 + i] = data->ctrl[i] - thrust;
      }
    }
  };

  inline void TransitionLocked(mjModel* model, mjData* data) override {
    // set mode to GUI selection
    if (mode > 0) {
      current_mode_ = mode - 1;
    } else {
      // goal position
      const double* goal_position = data->mocap_pos;

      // system's position
      double* position = SensorByName(model, data, "position");

      // position error
      double position_error[3];
      mju_sub3(position_error, position, goal_position);
      double position_error_norm = mju_norm3(position_error);

      if (position_error_norm <= 5.0e-1) {
        // update task state
        current_mode_ += 1;
        if (current_mode_ == model->nkey) {
          current_mode_ = 0;
        }
      }
    }

    // set goal
    mju_copy3(data->mocap_pos, model->key_mpos + 3 * current_mode_);
    mju_copy4(data->mocap_quat, model->key_mquat + 4 * current_mode_);
  }

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  int current_mode_ = 0;
  ResidualFn residual_;
};
}  // namespace mjpc

#endif  // MJPC_TASKS_QUADROTOR_QUADROTOR_H_
