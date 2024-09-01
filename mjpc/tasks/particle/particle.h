// Copyright 2022 DeepMind Technologies Limited
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

#ifndef MJPC_TASKS_PARTICLE_PARTICLE_H_
#define MJPC_TASKS_PARTICLE_PARTICLE_H_

#include <mujoco/mujoco.h>

#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>

#include "mjpc/planners/fabrics/include/fab_goal.h"
#include "mjpc/planners/fabrics/include/fab_math_util.h"
#include "mjpc/task.h"

// NOTE: Dynamic goal is not yet working for [Particle]
#define FAB_PARTICLE_DYNAMIC_GOAL_SUPPORTED (1)

// Dynamical Movement Primitives: Learning Attractor Models for Motor Behaviors
// https://ieeexplore.ieee.org/document/6797340
// https://homes.cs.washington.edu/~todorov/courses/amath579/reading/DynamicPrimitives.pdf
// https://studywolf.wordpress.com/2013/11/16/dynamic-movement-primitives-part-1-the-basics
// https://studywolf.wordpress.com/2016/05/13/dynamic-movement-primitives-part-4-avoiding-obstacles
namespace mjpc {
class Particle : public Task {
public:
  Particle() : residual_(this) {
    static_obstacles_num = 0;
    dynamic_obstacles_num = 10;
    actuator_kv = 1.f;
    first_joint_name_ = "root_x";
  }
  static double rand_val() { return FabRandom::rand<double>(-0.2, 0.2); }

  std::string Name() const override;
  std::string XmlPath() const override;
  std::string URDFPath() const override;
  std::string GetBaseBodyName() const override {
    static std::string name = "world";
    return name;
  }
  std::vector<std::string> GetEndtipNames() const override {
    static std::vector<std::string> names = {"base_link"};
    return names;
  }
  std::vector<std::string> GetCollisionLinkNames() const override {
    static std::vector<std::string> names = {"base_link"};
    return names;
  }
  FabLinkCollisionProps GetCollisionLinkProps() const override {
    static FabLinkCollisionProps props = {{GetCollisionLinkNames()[0], {0.01}}};
    return props;
  }
  int GetActionDim() const override { return 3; }
  std::vector<FabSubGoalPtr> GetSubGoals() const override {
    // NOTE: Due to base_link's fk having Z-translation as constant
    static const std::vector<int> indices = std::vector{0, 1};
    // NOTE: [Particle] & [ParticleFixed] can be toggled by users at run-time,
    // so [GetSubGoals()] are defined separately
    static auto subgoals = std::vector<FabSubGoalPtr>{
        std::make_shared<FabStaticSubGoal>(FabSubGoalConfig{.name = "subgoal0",
                                                            .type = FabSubGoalType::STATIC,
                                                            .is_primary_goal = true,
                                                            .epsilon = 0.1,
                                                            .indices = indices,
                                                            .weight = 5.0,
                                                            .parent_link_name = GetBaseBodyName(),
                                                            .child_link_name = GetEndtipNames()[0]})};
    auto& subgoal0_cfg = subgoals[0]->cfg_;
#if FAB_PARTICLE_DYNAMIC_GOAL_SUPPORTED
    if (!IsGoalFixed()) {
      subgoal0_cfg.type = FabSubGoalType::DYNAMIC;
    }
    auto& subgoal0_desired_state = subgoals[0]->cfg_.desired_state;
    subgoal0_desired_state = GetGoalState();
    if (subgoal0_desired_state.valid()) {
      const auto& pos = subgoal0_desired_state.pose.pos;
      const auto& vel = subgoal0_desired_state.linear_vel;
      const auto& acc = subgoal0_desired_state.linear_acc;
      subgoal0_desired_state.pose.pos = {pos[0], pos[1]};    // EXCLUDING pos[2]
      subgoal0_desired_state.linear_vel = {vel[0], vel[1]};  // EXCLUDING vel[2]
      subgoal0_desired_state.linear_acc = {acc[0], acc[1]};  // EXCLUDING vel[2]
    }
#else
    const auto* goal_pos = GetGoalPos();
    if (goal_pos) {
      subgoal0_cfg.desired_state.pose.pos = {goal_pos[0], goal_pos[1]};  // EXCLUDING goal_pos[2]
    }
#endif
    subgoal0_cfg.desired_state.pose_offset = FabPose{.pos = {0., 0., 0.}, .rot = {0., 0., 0.}};
    return subgoals;
  }

  bool IsGoalFixed() const override { return !FAB_PARTICLE_DYNAMIC_GOAL_SUPPORTED; }
  int GetDynamicObstaclesDimension() const override { return AreObstaclesFixed() ? 3 : 2; }
  int GetPlaneConstraintsNum() const override { return 1; }

  // NOTES on mutex:
  // Access to model & data: already locked by [sim.mtx]
  // Access to task local data: lock on [task_data_mutex_]
  int GetTargetObjectId() const override { return QueryBodyId("rigidmass"); }
  int GetTargetObjectGeomId() const override { return QueryGeomId("rigidmass"); }

  void SetGoalPos(const double* pos) const { SetBodyMocapPos("goal_mocap", pos); }
  const mjtNum* GetGoalPos() const override { return QueryBodyMocapPos("goal_mocap"); }
  const mjtNum* GetGoalVel() const override { return QueryBodyVel(QueryBodyId("goal")); }
  const mjtNum* GetGoalAcc() const override { return QueryBodyAcc(QueryBodyId("goal")); }

  static constexpr float PARTICLE_GOAL_REACH_THRESHOLD = 0.01;
  bool QueryGoalReached() override;
  void QueryObstacleStatesX() override;
  bool CheckBlocking(const double start[], const double end[]) override;

  class ResidualFn : public mjpc::BaseResidualFn {
  public:
    explicit ResidualFn(const Particle* task)
        : mjpc::BaseResidualFn(task), particle_task(dynamic_cast<const Particle*>(task_)) {}

    // -------- Residuals for particle task -------
    //   Number of residuals: 3
    //     Residual (0): position - goal_position
    //     Residual (1): velocity
    //     Residual (2): control
    // --------------------------------------------
    void Residual(const mjModel* model, const mjData* data, double* residual) const override;
    const Particle* particle_task = nullptr;
  };

  void TransitionLocked(mjModel* model, mjData* data) override;

  void ModifyScene(const mjModel* model, const mjData* data, mjvScene* scene) const override;

  virtual void MoveObstacles() {
    static constexpr bool ALTERNATING = 0;
    static int8_t sign = 1;
    if constexpr (ALTERNATING) {
      static bool sign_switched = true;
      if (fmod(data_->time, 2 * M_PI) < 0.1) {
        if (!sign_switched) {
          sign = -sign;
          sign_switched = true;
        }
      } else {
        sign_switched = false;
      }
    }
    for (auto i = 1; i <= GetTotalObstaclesNum(); ++i) {
      const auto angular_vel_i = (i > 3) ? (0.1 * i) : M_PI;
      const auto phase_i = angular_vel_i * data_->time;
      if constexpr (ALTERNATING) {
      } else {
        sign = (i % 2) == 0 ? 1 : -1;
      }
      const auto sin_pos_i = sign * mju_sin(phase_i);
      const auto cos_pos_i = mju_cos(phase_i);
      double obstacle_curve_pos[3] = {0.05 * log(i + 1) * sin_pos_i, 0.05 * log(i + 1) * cos_pos_i, 0.01};
      std::ostringstream obstacle_name;
      obstacle_name << "obstacle_" << (i - 1) << "_mocap";
      SetBodyMocapPos(obstacle_name.str().c_str(), obstacle_curve_pos);
    }
  }

  virtual void RandomizeObstacles() {
    for (auto i = 0; i < GetTotalObstaclesNum(); ++i) {
      std::ostringstream obstacle_name;
      obstacle_name << "obstacle_" << i << "_mocap";
      SetBodyMocapPos(obstacle_name.str().c_str(), (double[]){rand_val(), rand_val(), 0.01});
    }
  }

protected:
  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(residual_);
  }

  BaseResidualFn* InternalResidual() override { return &residual_; }

  bool IsFabricsSupported() const override { return true; }
  FabPlannerConfigPtr GetFabricsConfig() const override;

private:
  ResidualFn residual_;
};

// The same task, but the goal mocap body doesn't move.
class ParticleFixed : public Particle {
public:
  ParticleFixed() : residual_(this) {
    static_obstacles_num = 10;
    dynamic_obstacles_num = 0;
  }
  std::string Name() const override;
  std::string XmlPath() const override;

  bool IsGoalFixed() const override { return true; }
  std::vector<FabSubGoalPtr> GetSubGoals() const override {
    static const std::vector<int> indices = {0, 1};
    // [Particle] & [ParticleFixed] can be toggled by users at runtime
    static auto subgoals = std::vector<FabSubGoalPtr>{
        std::make_shared<FabStaticSubGoal>(FabSubGoalConfig{.name = "subgoal0",
                                                            .type = FabSubGoalType::STATIC,
                                                            .is_primary_goal = true,
                                                            .epsilon = 0.1,
                                                            .indices = indices,
                                                            .weight = 5.0,
                                                            .parent_link_name = GetBaseBodyName(),
                                                            .child_link_name = GetEndtipNames()[0]})};
    const auto* goal_pos = GetGoalPos();
    if (goal_pos) {
      subgoals[0]->cfg_.desired_state.pose.pos = {goal_pos[0], goal_pos[1]};  // EXCLUDING goal_pos[2]
    }
    subgoals[0]->cfg_.desired_state.pose_offset = FabPose{.pos = {0., 0., 0.}, .rot = {0., 0., 0.}};
    return subgoals;
  }

  class FixedResidualFn : public mjpc::BaseResidualFn {
  public:
    explicit FixedResidualFn(const ParticleFixed* task)
        : mjpc::BaseResidualFn(task), particle_fixed_task(dynamic_cast<const ParticleFixed*>(task_)) {}

    // -------- Residuals for particle task -------
    //   Number of residuals: 3
    //     Residual (0): position - goal_position
    //     Residual (1): velocity
    //     Residual (2): control
    // --------------------------------------------
    void Residual(const mjModel* model, const mjData* data, double* residual) const override;
    const ParticleFixed* particle_fixed_task = nullptr;
  };

  void TransitionLocked(mjModel* model, mjData* data) override;

protected:
  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<FixedResidualFn>(residual_);
  }

  BaseResidualFn* InternalResidual() override { return &residual_; }

private:
  FixedResidualFn residual_;
};
}  // namespace mjpc

#endif  // MJPC_TASKS_PARTICLE_PARTICLE_H_
