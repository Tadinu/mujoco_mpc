#pragma once

#include <mujoco/mujoco.h>

#include <memory>
#include <random>
#include <string>

#include "mjpc/planners/fabrics/include/fab_goal.h"
#include "mjpc/planners/fabrics/include/fab_planner.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
class PlanarRobot : public Task {
public:
  PlanarRobot() : residual_(this) {
    static_obstacles_num = 2;
    dynamic_obstacles_num = 0;
    actuator_kv = 1.f;
    first_joint_name_ = "joint1";
  }
  std::string Name() const override;
  std::string XmlPath() const override;
  std::string URDFPath() const override;
  std::string GetBaseBodyName() const override {
    static std::string name = "link0";
    return name;
  }
  std::vector<std::string> GetEndtipNames() const override {
    static std::vector<std::string> names = {"link3"};
    return names;
  }
  std::vector<std::string> GetCollisionLinkNames() const override {
    static std::vector<std::string> names = {"link2"};
    return names;
  }
  FabLinkCollisionProps GetCollisionLinkProps() const override {
    const auto& link_names = GetCollisionLinkNames();
    static FabLinkCollisionProps props = {{link_names[0], {0.2}}};
    return props;
  }
  int GetStaticObstaclesNum() const override { return 2; }
  int GetDynamicObstaclesNum() const override {
    return (planner_ && planner_->tuning_on_) ? static_cast<int>(GetCollisionLinkNames().size()) : 0;
  }
  int GetPlaneConstraintsNum() const override { return 0; }
  int GetActionDim() const override { return 2; }
  std::vector<FabSubGoalPtr> GetSubGoals() const override {
    // Static subgoals with static [desired_state.pos]
    static std::vector<FabSubGoalPtr> subgoals = {
        std::make_shared<FabStaticSubGoal>(
            FabSubGoalConfig{.name = "subgoal0",
                             .type = FabSubGoalType::STATIC,
                             .is_primary_goal = true,
                             .epsilon = 0.1,
                             .indices = {1, 2},
                             .weight = 1.0,
                             .parent_link_name = "link0",
                             .child_link_name = "link3",
                             .desired_state = {.pose = FabPose{.pos = {1.0, 1.2}}}}),
    };
    auto& subgoal0_cfg = subgoals[0]->cfg_;
    subgoal0_cfg.desired_state = GetGoalState();
    subgoal0_cfg.desired_state.pose_offset = FabPose{.pos = {0., 0., 0.}, .rot = {0., 0., 0.}};
    return subgoals;
  }

  bool IsGoalFixed() const override { return true; }
  int GetDynamicObstaclesDimension() const override { return 3; }
  std::vector<FabJointLimit> GetJointLimits() const override { return {}; }

  // NOTES on mutex:
  // Access to model & data: already locked by [sim.mtx]
  // Access to task local data: lock on [task_data_mutex_]
  int GetTargetObjectId() const override { return QueryBodyId("target"); }

  // Goals
  void SetGoalPos(const double* pos) const { SetBodyMocapPos("target_mocap", pos); }
  const mjtNum* GetGoalPos() const override { return QueryBodyMocapPos("target_mocap"); }
  const mjtNum* GetGoalVel() const override { return QueryTargetVel(); }
  const mjtNum* GetGoalAcc() const override { return QueryTargetAcc(); }

  void QueryObstacleStatesX() override;

  class ResidualFn : public mjpc::BaseResidualFn {
  public:
    explicit ResidualFn(const PlanarRobot* task) : mjpc::BaseResidualFn(task) {}

    void Residual(const mjModel* model, const mjData* data, double* residual) const override;

  private:
    friend class Bring;
  };

  void TransitionLocked(mjModel* model, mjData* data) override;
  void ResetLocked(const mjModel* model) override;

  void ModifyScene(const mjModel* model, const mjData* data, mjvScene* scene) const override {
    // Draw goal
    static constexpr float GREEN[] = {0.0, 1.0, 0.0, 1.0};
    mjpc::AddGeom(scene, mjGEOM_SPHERE, (mjtNum[]){0.02}, GetGoalPos(), /*mat=*/nullptr, GREEN);

    // Draw tip
    static constexpr float BLUE[] = {0.0, 0.0, 1.0, 1.0};
    double tip_pos[3];
    mju_copy(tip_pos, &data_->site_xpos[3 * mj_name2id(model, mjOBJ_SITE, "tip")], 3);
    mjpc::AddGeom(scene, mjGEOM_SPHERE, (mjtNum[]){0.02}, tip_pos, /*mat=*/nullptr, BLUE);
  }

protected:
  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(residual_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }
  bool IsFabricsSupported() const override { return true; }

private:
  ResidualFn residual_;
};
}  // namespace mjpc
