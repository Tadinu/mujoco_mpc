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

#ifndef MJPC_TASK_H_
#define MJPC_TASK_H_

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/norm.h"
#include "mjpc/planners/rmp/include/core/rmp_state.h"

namespace mjpc {

// tolerance for risk-neutral cost
inline constexpr double kRiskNeutralTolerance = 1.0e-6;

// maximum cost terms
inline constexpr int kMaxCostTerms = 128;

class Task;

// abstract class for a residual function
class AbstractResidualFn {
 public:
  virtual ~AbstractResidualFn() = default;

  virtual void Residual(const mjModel* model, const mjData* data,
                        double* residual) const = 0;
  virtual void CostTerms(double* terms, const double* residual,
                         bool weighted) const = 0;
  virtual double CostValue(const double* residual) const = 0;

  // copies weights and parameters from the Task instance. This should be
  // called from the Task class.
  virtual void Update() = 0;
};

// base implementation for ResidualFn implementations
class BaseResidualFn : public AbstractResidualFn {
 public:
  explicit BaseResidualFn(const Task* task);
  virtual ~BaseResidualFn() = default;

  void CostTerms(double* terms, const double* residual,
                 bool weighted) const override;
  double CostValue(const double* residual) const override;
  void Update() override;

 protected:
  int num_residual_;
  int num_term_;
  int num_trace_;
  std::vector<int> dim_norm_residual_;
  std::vector<int> num_norm_parameter_;
  std::vector<NormType> norm_;
  std::vector<double> weight_;
  std::vector<double> norm_parameter_;
  double risk_;
  std::vector<double> parameters_;
  const Task* task_;
};

// Thread-safe interface for classes that implement MJPC task specifications
class Task {
 public:
  // constructor
  Task() = default;
  virtual ~Task() = default;

  // delegates to ResidualLocked, while holding a lock
  std::unique_ptr<AbstractResidualFn> Residual() const;

  // ----- methods ----- //
  // calls Residual on the pointer returned from InternalResidual(), while
  // holding a lock
  void Residual(const mjModel* model, const mjData* data,
                double* residual) const;

  // Must be called whenever parameters or weights change outside Transition or
  // Reset, so that calls to Residual use the new parameters.
  // Calls InternalResidual()->Update() with a lock.
  void UpdateResidual();

  // Changes to data will affect the planner at the next set_state.  Changes to
  // model will only affect the physics and render threads, and will not affect
  // the planner. This is useful for studying planning under model discrepancy,
  // calls TransitionLocked and InternalResidual()->Update() while holding a
  // lock
  void Transition(mjModel* model, mjData* data);

  // get information from model
  // calls ResetLocked and InternalResidual()->Update() while holding a lock
  void Reset(const mjModel* model);

  // calls CostTerms on the pointer returned from InternalResidual(), while
  // holding a lock
  void CostTerms(double* terms, const double* residual) const;

  // calls CostTerms on the pointer returned from InternalResidual(), while
  // holding a lock
  void UnweightedCostTerms(double* terms, const double* residual) const;

  // calls CostValue on the pointer returned from InternalResidual(), while
  // holding a lock
  double CostValue(const double* residual) const;

  virtual void ModifyScene(const mjModel* model, const mjData* data,
                           mjvScene* scene) const {}

  virtual std::string Name() const = 0;
  virtual std::string XmlPath() const = 0;

  static constexpr int OBSTACLES_NUM = 10;
  virtual int GetTargetObjectId() const { return -1; }
  virtual int GetTargetObjectGeomId() const { return -1; }
  virtual bool CheckBlocking(const double start[], const double end[]) {return false;}

  // model
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  mjvScene* scene_ = nullptr;
  virtual const mjtNum* GetStartPos() { return nullptr;}
  virtual const mjtNum* GetStartVel() { return nullptr; }
  virtual const mjtNum* GetGoalPos() { return nullptr; }

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
      mju_copy3(&data->mocap_pos[3*bodyMocapId], pos);
      data->mocap_pos[3*bodyMocapId+2] = 0.01;
      //std::cout << body_name << ":" << bodyMocapId << " " << pos[0] << " " << pos[1] << std::endl;
    }
  }
  static mjtNum* QueryBodyMocapPos(const mjModel* model, const mjData* data,
                                   const char* body_name) {
    int bodyMocapId = GetBodyMocapId(model, body_name);
    return &data->mocap_pos[3*bodyMocapId];
  }

  void SetGeomColor(uint geom_id, const float* rgba) const {
    if (scene_ && (geom_id < model_->ngeom)) {
      memcpy(scene_->geoms[geom_id].rgba, rgba, sizeof(float) * 4);
    }
  }

  // mode
  int mode;

  // GUI toggles
  int reset = 0;
  int visualize = 0;

  // cost parameters
  int num_residual;
  int num_term;
  int num_trace;
  std::vector<int> dim_norm_residual;
  std::vector<int> num_norm_parameter;
  std::vector<NormType> norm;
  std::vector<double> weight;
  std::vector<double> norm_parameter;
  double risk;

  // residual parameters
  std::vector<double> parameters;
  std::array<mjtNum, 3 * OBSTACLES_NUM> ray_starts = {0.};
  std::array<mjtNum, 3 * OBSTACLES_NUM> ray_ends = {0.};
  bool last_goal_reached_ = false;

  // RMP
  using StateX = rmpcpp::State<3>;
  std::vector<StateX> obstacle_statesX_;

  // mutex which should be held on changes to data queried from mjdata
  mutable std::mutex task_data_mutex_;
  virtual void QueryObstacleStatesX() {}
  inline std::vector<StateX> GetObstacleStatesX() {
    std::vector<StateX> obstacle_statesX;
    {
      std::lock_guard<std::mutex> lock(task_data_mutex_);
      obstacle_statesX = obstacle_statesX_;
    }
    return obstacle_statesX;
  }

 protected:
  // returns a pointer to the ResidualFn instance that's used for physics
  // stepping and plotting, and is internal to the class
  virtual BaseResidualFn* InternalResidual() = 0;
  const BaseResidualFn* InternalResidual() const {
    return const_cast<Task*>(this)->InternalResidual();
  }
  // returns an object which can compute the residual function. the function
  // can assume that a lock on mutex_ is held when it's called
  virtual std::unique_ptr<AbstractResidualFn> ResidualLocked() const = 0;
  // implementation of Task::Transition() which can assume a lock is held.
  // in some cases the transition logic requires calling mj_forward (e.g., for
  // measuring contact forces), which will call the sensor callback, which calls
  // ResidualLocked. In order to avoid such resource contention, mutex_ might be
  // temporarily unlocked, but it must be locked again before returning.
  virtual void TransitionLocked(mjModel* model, mjData* data) {}
  // implementation of Task::Reset() which can assume a lock is held
  virtual void ResetLocked(const mjModel* model) {}
  // mutex which should be held on changes to InternalResidual.
  mutable std::mutex mutex_;

 private:
  // initial residual parameters from model
  void SetFeatureParameters(const mjModel* model);
};

}  // namespace mjpc

#endif  // MJPC_TASK_H_
