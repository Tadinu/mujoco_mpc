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

#ifndef MJPC_TASKS_ALLEGRO_ALLEGRO_H_
#define MJPC_TASKS_ALLEGRO_ALLEGRO_H_

#include <memory>
#include <string>

// mujoco
#include <mujoco/mujoco.h>

// mjpc
#include "mjpc/task.h"

// idto
#include "mjpc/planners/idto/idto_planner.h"
#include "mjpc/planners/idto/idto_yaml_config.h"
#include "mjpc/utilities.h"

namespace mjpc {
class Allegro : public Task {
public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public BaseResidualFn {
  public:
    explicit ResidualFn(const Allegro *task) : BaseResidualFn(task) {}
    void Residual(const mjModel *model, const mjData *data, double *residual) const override;
  };
  Allegro() : residual_(this) { idto_configs_path_ = GetModelPath("allegro/allegro_hand.yaml"); }

  void ResetLocked(const mjModel *model) override {
    // NOTE: THIS MUST RUN ON MAIN THREAD, TEMPORARILY PUT HERE
    auto *idto_planner = dynamic_cast<IdtoPlanner *>(planner_);
    if (idto_planner) {
      idto_planner->StartControl();
    }
  }

  // Reset the cube into the hand if it's on the floor
  void TransitionLocked(mjModel *model, mjData *data) override;

protected:
  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn *InternalResidual() override { return &residual_; }

  std::string target_type_name() const { return is_target_ball_ ? "ball" : "cube"; }
  std::string target_geom_name() const { return target_type_name(); }
  std::string target_body_name() const { return target_type_name(); }

private:
  ResidualFn residual_;
  bool is_target_ball_ = true;

  // =========================================================================================================
  // DRAKE IMPL --
  //
private:
  void InitMeshcat() override;
  void UpdateMeshcatFromIdtoConfigs() override;
  void CreateDrakePlantModel(drake::multibody::MultibodyPlant<double> *plant) const override;
};
}  // namespace mjpc

#endif  // MJPC_TASKS_ALLEGRO_ALLEGRO_H_
