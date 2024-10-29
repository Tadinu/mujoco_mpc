#pragma once

#include <mujoco/mujoco.h>

#include <memory>
#include <string>

#include "mjpc/task.h"

namespace mjpc {
class MPL : public Task {
public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public BaseResidualFn {
  public:
    explicit ResidualFn(const MPL* task) : BaseResidualFn(task) {}

    // ---------- Residuals for in-hand manipulation task ---------
    //   Number of residuals: 5
    //     Residual (0): cube_position - palm_position
    //     Residual (1): cube_orientation - cube_goal_orientation
    //     Residual (2): cube linear velocity
    //     Residual (3): cube angular velocity
    //     Residual (4): control
    // ------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data, double* residual) const override;
  };
  MPL() : residual_(this) {}

  // ----- Transition for in-hand manipulation task -----
  //   If cube is within tolerance or floor ->
  //   reset cube into hand.
  // -----------------------------------------------
  void TransitionLocked(mjModel* model, mjData* data) override;

protected:
  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

private:
  ResidualFn residual_;
};
}  // namespace mjpc
