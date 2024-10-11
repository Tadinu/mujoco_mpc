#pragma once
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
class AllegroX : public Task {
public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public BaseResidualFn {
  public:
    explicit ResidualFn(const AllegroX* task) : BaseResidualFn(task) {
    }

    void Residual(const mjModel* model, const mjData* data, double* residual) const override;
  };

  AllegroX() : residual_(this) { idto_configs_path_ = GetModelPath("allegro_x/allegro_hand.yaml"); }

  void ResetLocked(const mjModel* model) override {
    // NOTE: THIS MUST RUN ON MAIN THREAD, TEMPORARILY PUT HERE
    auto* idto_planner = dynamic_cast<IdtoPlanner*>(planner_);
    if (idto_planner) {
      idto_planner->StartControl();
    }
  }

  // Reset the cube into the hand if it's on the floor
  void TransitionLocked(mjModel* model, mjData* data) override;

protected:
  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }

  ResidualFn* InternalResidual() override { return &residual_; }

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
  void CreateDrakePlantModel(drake::multibody::MultibodyPlant<double>* plant) const override;
};
} // namespace mjpc
