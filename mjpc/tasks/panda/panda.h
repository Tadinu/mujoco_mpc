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

#ifndef MJPC_MJPC_TASKS_PANDA_PANDA_H_
#define MJPC_MJPC_TASKS_PANDA_PANDA_H_

#include <string>
#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
class Panda : public Task {
 public:
  Panda() : residual_(this) {}
  std::string Name() const override { return GetModelPath("panda/task.xml"); }
  std::string XmlPath() const override { return "Pick"; }
  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const Panda* task) : BaseResidualFn(task) {}
    inline void Residual(const mjModel* model, const mjData* data,
                        double* residual) const override {
      int counter = 0;

      // reach
      double* hand = SensorByName(model, data, "hand");
      double* box = SensorByName(model, data, "box");
      mju_sub3(residual + counter, hand, box);
      counter += 3;

      // bring
      double* box1 = SensorByName(model, data, "box1");
      double* target1 = SensorByName(model, data, "target1");
      mju_sub3(residual + counter, box1, target1);
      counter += 3;
      double* box2 = SensorByName(model, data, "box2");
      double* target2 = SensorByName(model, data, "target2");
      mju_sub3(residual + counter, box2, target2);
      counter += 3;

      // sensor dim sanity check
      // TODO: use this pattern everywhere and make this a utility function
      int user_sensor_dim = 0;
      for (int i = 0; i < model->nsensor; i++) {
        if (model->sensor_type[i] == mjSENS_USER) {
          user_sensor_dim += model->sensor_dim[i];
        }
      }
      if (user_sensor_dim != counter) {
        mju_error_i(
            "mismatch between total user-sensor dimension "
            "and actual length of residual %d",
            counter);
      }
    }
  };

  inline void TransitionLocked(mjModel* model, mjData* data) override
  {
    double residuals[100];
    residual_.Residual(model, data, residuals);
    double bring_dist = (mju_norm3(residuals+3) + mju_norm3(residuals+6)) / 2;

    // reset:
    if (data->time > 0 && bring_dist < .015) {
      // box:
      absl::BitGen gen_;
      data->qpos[0] = absl::Uniform<double>(gen_, -.5, .5);
      data->qpos[1] = absl::Uniform<double>(gen_, -.5, .5);
      data->qpos[2] = .05;

      // target:
      data->mocap_pos[0] = absl::Uniform<double>(gen_, -.5, .5);
      data->mocap_pos[1] = absl::Uniform<double>(gen_, -.5, .5);
      data->mocap_pos[2] = absl::Uniform<double>(gen_, .03, 1);
      data->mocap_quat[0] = absl::Uniform<double>(gen_, -1, 1);
      data->mocap_quat[1] = absl::Uniform<double>(gen_, -1, 1);
      data->mocap_quat[2] = absl::Uniform<double>(gen_, -1, 1);
      data->mocap_quat[3] = absl::Uniform<double>(gen_, -1, 1);
      mju_normalize4(data->mocap_quat);
    }
  }

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};
}  // namespace mjpc


#endif  // MJPC_MJPC_TASKS_PANDA_PANDA_H_
