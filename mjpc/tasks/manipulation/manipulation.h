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

#include <mujoco/mujoco.h>

#include "mjpc/task.h"
#include "mjpc/tasks/manipulation/common.h"
#include "mjpc/utilities.h"

namespace mjpc::manipulation {
class Bring : public Task {
public:
  Bring() : residual_(this, ModelValues()) { actuator_kv = 1.f; }
  std::string Name() const override;
  std::string XmlPath() const override;
  std::string URDFPath() const override;
  std::string GetBaseBodyName() const override {
    static std::string name = "panda_link0";
    return name;
  }
  std::vector<std::string> GetEndtipNames() const override {
    static std::vector<std::string> names = {"panda_leftfinger"};
    return names;
  }
  std::vector<std::string> GetCollisionLinkNames() const override {
    static std::vector<std::string> names = {"panda_hand", "panda_link3", "panda_link4"};
    return names;
  }
  FabLinkCollisionProps GetCollisionLinkProps() const override {
    const auto& link_names = GetCollisionLinkNames();
    static FabLinkCollisionProps props = {
        {link_names[0], {0.02}}, {link_names[1], {0.02}}, {link_names[2], {0.02}}};
    return props;
  }
  FabSelfCollisionNamePairs GetSelfCollisionNamePairs() const override {
    return {{"panda_hand", {"panda_link2", "panda_link4"}}};
  }
  int GetDynamicObstaclesNum() const override { return static_cast<int>(GetCollisionLinkNames().size()); }
  int GetPlaneConstraintsNum() const override { return 0; }

  int GetActionDim() const override { return 7; }
  std::vector<FabSubGoalPtr> GetSubGoals() const override {
    // Static subgoals with static [desired_position]
    static std::vector<FabSubGoalPtr> subgoals = {
        std::make_shared<FabStaticSubGoal>(FabSubGoalConfig{.name = "subgoal0",
                                                            .type = FabSubGoalType::STATIC,
                                                            .is_primary_goal = true,
                                                            .epsilon = 0.05,
                                                            .indices = {0, 1, 2},
                                                            .weight = 0.7,
                                                            .parent_link_name = "panda_link0",
                                                            .child_link_name = "panda_hand",
                                                            .desired_position = {0.6, 0., 0.4}}),

        std::make_shared<FabStaticSubGoal>(
            FabSubGoalConfig{.name = "subgoal1",
                             .type = FabSubGoalType::STATIC,
                             .is_primary_goal = false,
                             .epsilon = 0.05,
                             .indices = {0, 1, 2},
                             .weight = 6.0,
                             .parent_link_name = "panda_link7",
                             .child_link_name = "panda_hand",
                             .desired_position = {6.55186038e-18, 0.00000000e+00, -1.07000000e-01}}),

        std::make_shared<FabStaticJointSpaceSubGoal>(
            FabSubGoalConfig{.name = "subgoal2",
                             .type = FabSubGoalType::STATIC_JOINT_SPACE,
                             .is_primary_goal = false,
                             .epsilon = 0.05,
                             .indices = {6},
                             .weight = 6.0,
                             .parent_link_name = "panda_link0",
                             .child_link_name = "panda_hand",
                             .desired_position = {M_PI_4}})};

    const auto* goal_pos = GetGoalPos();
    mju_copy(subgoals[0]->cfg_.desired_position.data(), goal_pos, 3);
    mju_copy(subgoals[1]->cfg_.desired_position.data(), goal_pos, 3);
    return subgoals;
  }

  bool IsGoalFixed() const override { return true; }
  int GetDynamicObstaclesDim() const override { return 3; }
  std::vector<FabJointLimit> GetJointLimits() const override {
    return {{-2.8973, 2.8973},   // panda_joint1
            {-1.7628, 1.7628},   // panda_joint2
            {-2.8973, 2.8973},   // panda_joint3
            {-3.0718, -0.0698},  // panda_joint4
            {-2.8973, 2.8973},   // panda_joint5
            {-0.0175, 3.7525},   // panda_joint6
            {-2.8973, 2.8973}};  // panda_joint7
  }

  // NOTES on mutex:
  // Access to model & data: already locked by [sim.mtx]
  // Access to task local data: lock on [task_data_mutex_]
  int GetTargetObjectId() const override { return mj_name2id(model_, mjOBJ_BODY, "target"); }

  const mjtNum* QueryTargetPos() const {
    if (model_) {
      return &data_->xipos[3 * GetTargetObjectId()];
    }
    return nullptr;
  }

  const mjtNum* QueryTargetVel() const {
    if (model_) {
      static double lvel[3] = {0};
      auto target_id = GetTargetObjectId();
#if 1
      memcpy(lvel, &data_->cvel[6 * target_id + 3], sizeof(mjtNum) * 3);
#else
      mjtNum vel[6];
      mj_objectVelocity(model_, data_, mjOBJ_BODY, target_id, vel, 0);
      memcpy(lvel, &vel[3], sizeof(mjtNum) * 3);
#endif
      return &lvel[0];
    }
    return nullptr;
  }

  const mjtNum* GetStartPos() override { return QueryTargetPos(); }
  const mjtNum* GetStartVel() override { return QueryTargetVel(); }
  const mjtNum* GetGoalPos() const override { return QueryTargetPos(); }

  bool QueryGoalReached() override { return false; }
  void QueryObstacleStatesX() override {}

  std::vector<double> QueryJointPos(int dof) const override {
    if (model_ && data_) {
      std::vector<double> qpos(dof, 0);
      memcpy(qpos.data(), data_->qpos + model_->jnt_qposadr[mj_name2id(model_, mjOBJ_JOINT, "joint1")],
             std::min(model_->nq, dof) * sizeof(double));
      return qpos;
    }
    return {};
  }

  std::vector<double> QueryJointVel(int dof) const override {
    if (model_ && data_) {
      std::vector<double> qvel(dof, 0);
      memcpy(qvel.data(), data_->qvel + model_->jnt_dofadr[mj_name2id(model_, mjOBJ_JOINT, "joint1")],
             std::min(model_->nv, dof) * sizeof(double));
      return qvel;
    }
    return {};
  }

  class ResidualFn : public mjpc::BaseResidualFn {
  public:
    explicit ResidualFn(const Bring* task, ModelValues values)
        : mjpc::BaseResidualFn(task), model_vals_(std::move(values)) {}

    void Residual(const mjModel* model, const mjData* data, double* residual) const override;

  private:
    friend class Bring;
    ModelValues model_vals_;
  };

  void TransitionLocked(mjModel* model, mjData* data) override;
  void ResetLocked(const mjModel* model) override;

  void ModifyScene(const mjModel* model, const mjData* data, mjvScene* scene) const override {
    // Draw goal
    static constexpr float GREEN[] = {0.0, 1.0, 0.0, 1.0};
    mjpc::AddGeom(scene, mjGEOM_SPHERE, (mjtNum[]){0.05}, GetGoalPos(), /*mat=*/nullptr, GREEN);

    // Draw pinch
    static constexpr float BLUE[] = {0.0, 0.0, 1.0, 1.0};
    double pinch_pos[3];
    mju_copy(pinch_pos, &data_->site_xpos[3 * mj_name2id(model, mjOBJ_SITE, "pinch")], 3);
    mjpc::AddGeom(scene, mjGEOM_SPHERE, (mjtNum[]){0.02}, pinch_pos, /*mat=*/nullptr, BLUE);
  }

protected:
  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.model_vals_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }
  bool IsFabricsSupported() const override { return true; }
  FabPlannerConfigPtr GetFabricsConfig(bool is_static_env) const override;

private:
  ResidualFn residual_;
};
}  // namespace mjpc::manipulation

#endif  // MJPC_MJPC_TASKS_MANIPULATION_MANIPULATION_H_
