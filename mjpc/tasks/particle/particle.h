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
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class Particle : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  const double* GetStartPos() const override {
   if (model_) {
#if 0 // Unclear why xipos(mjOBJ_BODY) xpos(mjOBJ_BODY) & geom_xpos are not correct???
    return &data_->xipos[mj_name2id(model_, mjOBJ_BODY, "pointmass")];
    return &data_->geom_xpos[mj_name2id(model_, mjOBJ_GEOM, "pointmass")];
#endif
    int site_start = mj_name2id(model_, mjOBJ_SITE, "tip");
    return &data_->site_xpos[site_start];
   }
   return nullptr;
  }
  const double* GetStartVel() const override {
    if (model_) {
      static double lvel[3] = {0};
      auto pointmass_id = mj_name2id(model_, mjOBJ_BODY, "pointmass");
#if 1
      memcpy(lvel, &data_->cvel[6*pointmass_id+3], sizeof(mjtNum) * 3);
#else
      mjtNum vel[6];
      mj_objectVelocity(model_, data_, mjOBJ_BODY, pointmass_id, vel, 0);
      memcpy(lvel, &vel[3], sizeof(mjtNum) * 3);
#endif
      return &lvel[0];
    }
    return nullptr;
  }
  const double* GetGoalPos() const override {
   if (model_) {
    int goal = mj_name2id(model_, mjOBJ_BODY, "goal");
    return &data_->mocap_pos[model_->body_mocapid[goal]];
   }
   return nullptr;
  }
  bool checkCollision(double pos[]) const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Particle* task) : mjpc::BaseResidualFn(task) {}
    // -------- Residuals for particle task -------
    //   Number of residuals: 3
    //     Residual (0): position - goal_position
    //     Residual (1): velocity
    //     Residual (2): control
    // --------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };
  Particle() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

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
        : mjpc::BaseResidualFn(task) {}
    // -------- Residuals for particle task -------
    //   Number of residuals: 3
    //     Residual (0): position - goal_position
    //     Residual (1): velocity
    //     Residual (2): control
    // --------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };
  ParticleFixed() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override {}
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
