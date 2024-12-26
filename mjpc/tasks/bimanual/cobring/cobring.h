#pragma once

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc::panda {
class CoBring : public Task {
public:
  std::string Name() const override;
  std::string XmlPath() const override;
  bool IsBimanualSupported() const override { return true; }

  std::string GetBaseBodyName() const override {
    return "torso";
  }

  class ResidualFn : public BaseResidualFn {
  public:
    explicit ResidualFn(const CoBring* task) : BaseResidualFn(task) {
    }

    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  CoBring() : residual_(this) {
  }

  void TransitionLocked(mjModel* model, mjData* data) override;

  int GetTargetObjectId() const override { return QueryBodyId("target"); }
  const mjtNum* GetGoalPos() const override { return QueryBodyMocapPos("target"); }
  const mjtNum* GetGoalVel() const override { return QueryTargetVel(); }
  const mjtNum* GetGoalAcc() const override { return QueryTargetAcc(); }

  std::vector<FabSubGoalPtr> GetSubGoals() const override {
    // Static subgoals with static [desired_state.pos]
    static std::vector<FabSubGoalPtr> subgoals = {std::make_shared<FabStaticSubGoal>(
                                                      FabSubGoalConfig{.name = "subgoal0",
                                                                       .type = FabSubGoalType::STATIC,
                                                                       .is_primary_goal = true,
                                                                       .epsilon = 0.02,
                                                                       .indices = {0, 1, 2},
                                                                       .weight = 0.4,
                                                                       .parent_link_name = "panda0_link0",
                                                                       .child_link_name =
                                                                       "panda0_leftfinger"}),
                                                  std::make_shared<FabStaticSubGoal>(
                                                      FabSubGoalConfig{.name = "subgoal1",
                                                                       .type = FabSubGoalType::STATIC,
                                                                       .is_primary_goal = true,
                                                                       .epsilon = 0.02,
                                                                       .indices = {0, 1, 2},
                                                                       .weight = 0.4,
                                                                       .parent_link_name = "panda1_link0",
                                                                       .child_link_name =
                                                                       "panda1_leftfinger"})
    };
    auto& subgoal0_cfg = subgoals[0]->cfg_;
    if (!IsGoalFixed()) {
      subgoal0_cfg.type = FabSubGoalType::DYNAMIC;
    }
    subgoal0_cfg.desired_state = GetGoalState();
    if (subgoal0_cfg.desired_state.pose.empty()) {
      subgoal0_cfg.desired_state.reset();
    }
    // Take goal's rot as the desired_state's rot offset
    subgoal0_cfg.desired_state.pose_offset =
        FabPose{.pos = {0., 0., 0.}, .rot = subgoal0_cfg.desired_state.pose.rot};
    return subgoals;
  }

protected:
  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }

  ResidualFn* InternalResidual() override { return &residual_; }
  double last_solve_time = 0;

private:
  ResidualFn residual_;
};
} // namespace mjpc
