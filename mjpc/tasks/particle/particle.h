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

#include "mjpc/task.h"

// Dynamical Movement Primitives: Learning Attractor Models for Motor Behaviors
// https://ieeexplore.ieee.org/document/6797340
// https://homes.cs.washington.edu/~todorov/courses/amath579/reading/DynamicPrimitives.pdf
// https://studywolf.wordpress.com/2013/11/16/dynamic-movement-primitives-part-1-the-basics
// https://studywolf.wordpress.com/2016/05/13/dynamic-movement-primitives-part-4-avoiding-obstacles
namespace mjpc {
class Particle : public Task {
public:
  Particle() : residual_(this) {}
  std::random_device rd;                  // To obtain a seed for the random number engine
  std::mt19937 gen = std::mt19937(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis = std::uniform_real_distribution<>(-0.2, 0.2);
  double rand_val() { return dis(gen); }

  std::string Name() const override;
  std::string XmlPath() const override;

  bool IsGoalFixed() const override { return !FAB_DYNAMIC_GOAL_SUPPORTED; }
  bool AreObstaclesFixed() const override { return false; }

  // NOTES on mutex:
  // Access to model & data: already locked by [sim.mtx]
  // Access to task local data: lock on [task_data_mutex_]
  int GetTargetObjectId() const override { return mj_name2id(model_, mjOBJ_BODY, "rigidmass"); }
  int GetTargetObjectGeomId() const override { return mj_name2id(model_, mjOBJ_GEOM, "rigidmass"); }

  const mjtNum* QueryParticlePos() const {
    if (model_) {
#if 1
      return &data_->xipos[3 * GetTargetObjectId()];
#else
      int site_start = mj_name2id(model_, mjOBJ_SITE, "tip");
      return &data_->site_xpos[site_start];
#endif
    }
    return nullptr;
  }

  const mjtNum* QueryParticleVel() const {
    if (model_) {
      static double lvel[3] = {0};
      auto rigidmass_id = GetTargetObjectId();
#if 1
      memcpy(lvel, &data_->cvel[6 * rigidmass_id + 3], sizeof(mjtNum) * 3);
#else
      mjtNum vel[6];
      mj_objectVelocity(model_, data_, mjOBJ_BODY, rigidmass_id, vel, 0);
      memcpy(lvel, &vel[3], sizeof(mjtNum) * 3);
#endif
      return &lvel[0];
    }
    return nullptr;
  }

  mjtNum particle_pos_[3];

  mjtNum* GetParticlePos() {
    MJPC_LOCK_TASK_DATA_ACCESS;
    mju_copy3(particle_pos_, QueryParticlePos());
    return &particle_pos_[0];
  }

  mjtNum start_pos_[3];

  const mjtNum* GetStartPos() override {
    MJPC_LOCK_TASK_DATA_ACCESS;
    mju_copy3(start_pos_, QueryParticlePos());
    return &start_pos_[0];
  }

  mjtNum start_vel_[3];

  const mjtNum* GetStartVel() override {
    MJPC_LOCK_TASK_DATA_ACCESS;
    mju_copy3(start_vel_, QueryParticleVel());
    return &start_vel_[0];
  }

  std::vector<double> QueryJointPos(int dof) const override {
    if (model_ && data_ && (model_->nq <= dof)) {
      std::vector<double> qpos(dof);
      memcpy(qpos.data(), &data_->qpos[0], model_->nq * sizeof(mjtNum));
      return qpos;
    }
    return {};
  }

  std::vector<double> QueryJointVel(int dof) const override {
    if (model_ && data_ && (model_->nv <= dof)) {
      std::vector<double> qvel(dof);
      memcpy(qvel.data(), &data_->qvel[0], model_->nv * sizeof(mjtNum));
      return qvel;
    }
    return {};
  }

  void SetGoalPos(const double* pos) { SetBodyMocapPos("goal", pos); }

  mjtNum goal_pos_[3];

  const mjtNum* GetGoalPos() override {
    MJPC_LOCK_TASK_DATA_ACCESS;
    mju_copy3(goal_pos_, QueryGoalPos());
    return &goal_pos_[0];
  }

  const mjtNum* QueryGoalPos() const { return QueryBodyMocapPos("goal"); }
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

  virtual void ModifyScene(const mjModel* model, const mjData* data, mjvScene* scene) const override;

  virtual void MoveObstacles() {
    for (auto i = 1; i < (OBSTACLES_NUM + 1); ++i) {
      double obstacle_curve_pos[2] = {0.05 * log(i + 1) * mju_sin(0.2 * i * data_->time),
                                      0.05 * log(i + 1) * mju_cos(0.2 * i * data_->time)};
      std::ostringstream obstacle_name;
      obstacle_name << "obstacle_" << (i - 1);
      SetBodyMocapPos(obstacle_name.str().c_str(), obstacle_curve_pos);
    }
  }

  virtual void RandomizeObstacles() {
    for (auto i = 1; i < (OBSTACLES_NUM + 1); ++i) {
      std::ostringstream obstacle_name;
      obstacle_name << "obstacle_" << (i - 1);
      SetBodyMocapPos(obstacle_name.str().c_str(), (double[]){rand_val(), rand_val()});
    }
  }

protected:
  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }

  BaseResidualFn* InternalResidual() override { return &residual_; }

  FabPlannerConfig GetFabricsConfig(bool is_static_env) const override;

private:
  ResidualFn residual_;
};

// The same task, but the goal mocap body doesn't move.
class ParticleFixed : public Particle {
public:
  ParticleFixed() : residual_(this) {}
  std::string Name() const override;
  std::string XmlPath() const override;

  bool IsGoalFixed() const override { return true; }
  bool AreObstaclesFixed() const override { return true; }

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
    return std::make_unique<FixedResidualFn>(this);
  }

  BaseResidualFn* InternalResidual() override { return &residual_; }

private:
  FixedResidualFn residual_;
};
}  // namespace mjpc

#endif  // MJPC_TASKS_PARTICLE_PARTICLE_H_
