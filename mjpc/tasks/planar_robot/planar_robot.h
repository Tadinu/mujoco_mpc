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
  std::string RobotModelPath() const override;
  std::string GetBaseBodyName() const override { return "link0"; }
  std::vector<std::string> GetEndtipNames() const override {
    return {(GetActionDim() == 4) ? "finger" : "hand"};
  }
  std::vector<std::string> GetCollisionLinkNames() const override {
    return (GetActionDim() == 4) ? std::vector<std::string>{"link2", "hand", "finger"}
                                 : std::vector<std::string>{"link2", "hand"};
  }
  FabLinkCollisionProps GetCollisionLinkProps() const override {
    static FabLinkCollisionProps props;
    if (props.empty()) {
      for (const auto& link_name : GetCollisionLinkNames()) {
        props[link_name] = {QueryGeomSizeMax(link_name.c_str())};
      }
    }
    return props;
  }
  int GetStaticObstaclesNum() const override { return 2; }
  int GetDynamicObstaclesNum() const override {
    return (planner_ && planner_->is_tuning_on()) ? static_cast<int>(GetCollisionLinkNames().size()) : 0;
  }
  int GetPlaneConstraintsNum() const override { return 0; }
  int GetActionDim() const override {
    return RobotModelPath().ends_with("4dof.xml")   ? 4
           : RobotModelPath().ends_with("3dof.xml") ? 3
           : RobotModelPath().ends_with("2dof.xml") ? 2
                                                    : 0;
  }
  std::vector<FabSubGoalPtr> GetSubGoals() const override {
    // Static subgoals with static [desired_state.pos]
    static std::vector<FabSubGoalPtr> subgoals = {
        std::make_shared<FabStaticSubGoal>(FabSubGoalConfig{
            .name = "subgoal0",
            .type = FabSubGoalType::STATIC,
            .is_primary_goal = true,
            .epsilon = 0.1,
            // NOTE: For [planar_2dof] or singular-axis robots in general, due to
            // zeroed-out elements in fk, only a subset of goal indices is used
            .indices = (GetActionDim() == 2) ? std::vector<int>{1, 2} : std::vector<int>{0, 1, 2},
            .weight = 1.0,
            .parent_link_name = "link0",
            .child_link_name = (GetActionDim() == 4) ? "finger" : "hand",
        }),
    };
    auto& subgoal0_cfg = subgoals[0]->cfg_;
    subgoal0_cfg.desired_state = GetGoalState();
    if (subgoal0_cfg.desired_state.pose.empty()) {
      subgoal0_cfg.desired_state.reset();
    }
    subgoal0_cfg.desired_state.pose_offset = FabPose::zeros(3);
    return subgoals;
  }

  bool AreObstaclesFixed() const override { return true; }
  bool IsGoalFixed() const override { return true; }
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
    mju_copy(tip_pos, &data_->site_xpos[3 * mj_name2id(model, mjOBJ_SITE, "hand")], 3);
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
