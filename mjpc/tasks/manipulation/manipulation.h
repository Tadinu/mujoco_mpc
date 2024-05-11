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

#ifndef MJPC_MJPC_TASKS_MANIPULATION_MANIPULATION_H_
#define MJPC_MJPC_TASKS_MANIPULATION_MANIPULATION_H_

#include <string>

#include <absl/random/random.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"
#include "mjpc/task.h"
#include "mjpc/tasks/manipulation/common.h"

namespace mjpc::manipulation {
class Bring : public Task {
 public:
  inline std::string XmlPath() const override {
    return GetModelPath("manipulation/task_panda_bring.xml");
  }
  inline std::string Name() const override { return "PickAndPlace"; }

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Bring* task, ModelValues values)
        : mjpc::BaseResidualFn(task), model_vals_(std::move(values)) {}

    void Residual(const mjModel* model,
                  const mjData* data,
                  double* residual) const override {
      int counter = 0;

      // reach
      double hand[3] = {0};
      ComputeRobotiqHandPos(model, data, model_vals_, hand);

      double* object = SensorByName(model, data, "object");
      mju_sub3(residual + counter, hand, object);
      counter += 3;

      // bring
      for (int i=0; i < 8; i++) {
        double* object_i = SensorByName(model, data, std::to_string(i).c_str());
        double* target_i = SensorByName(model, data,
                                        (std::to_string(i) + "t").c_str());
        residual[counter++] = mju_dist3(object_i, target_i);
      }

      // careful
      int object_id = mj_name2id(model, mjOBJ_BODY, "object");
      residual[counter++] = CarefulCost(model, data, model_vals_, object_id);

      // away
      residual[counter++] = mju_min(0, hand[2] - 0.6);


      // sensor dim sanity check
      CheckSensorDim(model, counter);
    }

   private:
    friend class Bring;
    ModelValues model_vals_;
  };

  Bring() : residual_(this, ModelValues()) {}
  void TransitionLocked(mjModel* model, mjData* data) override {
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
      data->qpos[7+0] = 0.45;
      data->qpos[7+1] = 0;
      data->qpos[7+2] = 0.15;
      data->qpos[7+3] = absl::Uniform<double>(gen_, -1, 1);
      data->qpos[7+4] = absl::Uniform<double>(gen_, -1, 1);
      data->qpos[7+5] = absl::Uniform<double>(gen_, -1, 1);
      data->qpos[7+6] = absl::Uniform<double>(gen_, -1, 1);
      mju_normalize4(data->qpos + 13);

      // return stage: bring
      data->userdata[0] = 0;
    }
  }

  void ResetLocked(const mjModel* model) override {
    residual_.model_vals_ = ModelValues::FromModel(model);
  }
 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.model_vals_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};
}  // namespace mjpc::manipulation


#endif  // MJPC_MJPC_TASKS_MANIPULATION_MANIPULATION_H_
