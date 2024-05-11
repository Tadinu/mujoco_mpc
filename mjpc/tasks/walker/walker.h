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

#ifndef MJPC_TASKS_WALKER_WALKER_H_
#define MJPC_TASKS_WALKER_WALKER_H_

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
class Walker : public Task {
 public:
  Walker() : residual_(this) {}
  inline std::string XmlPath() const override {
    return GetModelPath("walker/task.xml");
  }
  inline std::string Name() const override { return "Walker"; }

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Walker* task) : mjpc::BaseResidualFn(task) {}

// --------- Residuals for walker task --------
//   Number of residuals: 4
//     Residual (0): control
//     Residual (1): position_z - height_goal
//     Residual (2): body_z_axis - 1.0
//     Residual (3): velocity_x - speed_goal
//   Parameters: 2
//     Parameter (0): height_goal
//     Parameter (1): speed_goal
// --------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override {
      int counter = 0;
      // ---------- Residual (0) ----------
      mju_copy(&residual[counter], data->ctrl, model->nu);
      counter += model->nu;

      // ---------- Residual (1) -----------
      double height = SensorByName(model, data, "torso_position")[2];
      residual[counter++] = height - parameters_[0];

      // ---------- Residual (2) ----------
      double torso_up = SensorByName(model, data, "torso_zaxis")[2];
      residual[counter++] = torso_up - 1.0;

      // ---------- Residual (3) ----------
      double com_vel = SensorByName(model, data, "torso_subtreelinvel")[0];
      residual[counter++] = com_vel - parameters_[1];

      // sensor dim sanity check
      // TODO: use this pattern everywhere and make this a utility function
      int user_sensor_dim = 0;
      for (int i=0; i < model->nsensor; i++) {
        if (model->sensor_type[i] == mjSENS_USER) {
          user_sensor_dim += model->sensor_dim[i];
        }
      }
      if (user_sensor_dim != counter) {
        mju_error_i("mismatch between total user-sensor dimension "
                    "and actual length of residual %d", counter);
      }
    }
  };

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;

};
}  // namespace mjpc

#endif  // MJPC_TASKS_WALKER_WALKER_H_
