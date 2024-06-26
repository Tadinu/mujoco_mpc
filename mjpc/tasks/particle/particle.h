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

#include <memory>
#include <string>
#include <iostream>
#include <sstream>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

// Dynamical Movement Primitives: Learning Attractor Models for Motor Behaviors
// https://ieeexplore.ieee.org/document/6797340
// https://homes.cs.washington.edu/~todorov/courses/amath579/reading/DynamicPrimitives.pdf
// https://studywolf.wordpress.com/2013/11/16/dynamic-movement-primitives-part-1-the-basics
// https://studywolf.wordpress.com/2016/05/13/dynamic-movement-primitives-part-4-avoiding-obstacles
namespace mjpc {
class Particle : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  virtual int GetTargetObjectId() const override { return mj_name2id(model_, mjOBJ_BODY, "rigidmass"); }
  const double* GetStartPos() const override {
   if (model_) {
#if 1
    return &data_->xpos[3* GetTargetObjectId()];
#else
    int site_start = mj_name2id(model_, mjOBJ_SITE, "tip");
    return &data_->site_xpos[site_start];
#endif
   }
   return nullptr;
  }
  const double* GetStartVel() const override {
    if (model_) {
      static double lvel[3] = {0};
      auto rigidmass_id = GetTargetObjectId();
#if 1
      memcpy(lvel, &data_->cvel[6*rigidmass_id+3], sizeof(mjtNum) * 3);
#else
      mjtNum vel[6];
      mj_objectVelocity(model_, data_, mjOBJ_BODY, pointmass_id, vel, 0);
      memcpy(lvel, &vel[3], sizeof(mjtNum) * 3);
#endif
      return &lvel[0];
    }
    return nullptr;
  }
  static int GetBodyMocapId(const mjModel* model, const char* body_name) {
    if (model) {
      int goal = mj_name2id(model, mjOBJ_BODY, body_name);
      return model->body_mocapid[goal];
    }
    return -1;
  }

  static void SetBodyMocapPos(const mjModel* model, const mjData* data,
                              const char* body_name, const double* pos) {
    if (model && data) {
      int bodyMocapId = GetBodyMocapId(model, body_name);
      data->mocap_pos[3*bodyMocapId] = pos[0]; // x
      data->mocap_pos[3*bodyMocapId + 1] = pos[1]; // y
      //data->mocap_pos[3*bodyMocapId + 2] = 0.1; // z
      //std::cout << body_name << ":" << bodyMocapId << " " << pos[0] << " " << pos[1] << std::endl;
    }
  }
  static double* GetBodyMocapPos(const mjModel* model, const mjData* data,
                                 const char* body_name) {
    int bodyMocapId = GetBodyMocapId(model, body_name);
    static double pos[2] = {0};
    if (data) {
      pos[0] = data->mocap_pos[bodyMocapId],
      pos[1] = data->mocap_pos[bodyMocapId + 1];
    }
    return pos;
  }
  void SetGoalPos(const double* pos) {
    SetBodyMocapPos(model_, data_, "goal", pos);
  }
  const double* GetGoalPos() const override {
    return GetBodyMocapPos(model_, data_, "goal");
  }
  bool CheckBlocking(const double start[], const double end[]) override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Particle* task)
         : mjpc::BaseResidualFn(task),
           particle_task(dynamic_cast<const Particle*>(task_)) {
    }
    // -------- Residuals for particle task -------
    //   Number of residuals: 3
    //     Residual (0): position - goal_position
    //     Residual (1): velocity
    //     Residual (2): control
    // --------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
    const Particle* particle_task = nullptr;
  };
  Particle() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

  virtual void MoveObstacles() {
    for(auto i = 1; i < (OBSTACLES_NUM+1); ++i) {
      double obstacle_curve_pos[2] = {0.05 * log(i+1) * mju_sin(0.2*i * data_->time),
                                      0.05 * log(i+1) * mju_cos(0.2*i * data_->time)};
      std::ostringstream obstacle_name;
      obstacle_name << "obstacle_" << (i - 1);
      SetBodyMocapPos(model_, data_, obstacle_name.str().c_str(), obstacle_curve_pos);
    }
  }
 protected:
  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  BaseResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

// The same task, but the goal mocap body doesn't move.
class ParticleFixed : public Particle {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class FixedResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit FixedResidualFn(const ParticleFixed* task)
         : mjpc::BaseResidualFn(task),
           particle_fixed_task(dynamic_cast<const ParticleFixed*>(task_)) {
    }
    // -------- Residuals for particle task -------
    //   Number of residuals: 3
    //     Residual (0): position - goal_position
    //     Residual (1): velocity
    //     Residual (2): control
    // --------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
    const ParticleFixed* particle_fixed_task = nullptr;
  };
  ParticleFixed() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override {
    //MoveObstacles();
  }
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
