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

#include <mujoco/mujoco.h>

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "mjpc/norm.h"
#include "mjpc/planners/fabrics/include/fab_config.h"
#include "mjpc/planners/fabrics/include/fab_goal.h"
#include "mjpc/planners/rmp/include/core/rmp_state.h"

namespace mjpc {
// tolerance for risk-neutral cost
inline constexpr double kRiskNeutralTolerance = 1.0e-6;

// maximum cost terms
inline constexpr int kMaxCostTerms = 128;

class Task;
class Planner;

#define MJPC_LOCK_TASK_DATA_ACCESS std::lock_guard<std::mutex> lock0(task_data_mutex_);

// abstract class for a residual function
class AbstractResidualFn {
public:
  virtual ~AbstractResidualFn() = default;

  virtual void Residual(const mjModel* model, const mjData* data, double* residual) const = 0;
  virtual void CostTerms(double* terms, const double* residual, bool weighted) const = 0;
  virtual double CostValue(const double* residual) const = 0;

  // copies weights and parameters from the Task instance. This should be
  // called from the Task class.
  virtual void Update() = 0;
};

// base implementation for ResidualFn implementations
class BaseResidualFn : public AbstractResidualFn {
public:
  explicit BaseResidualFn(const Task* task);
  ~BaseResidualFn() override = default;

  void CostTerms(double* terms, const double* residual, bool weighted) const override;
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

  // Fabrics config
  virtual bool IsFabricsSupported() const { return false; }

  virtual FabPlannerConfigPtr GetFabricsConfig() const { return std::make_shared<FabPlannerConfig>(); }

  // delegates to ResidualLocked, while holding a lock
  std::unique_ptr<AbstractResidualFn> Residual() const;

  // ----- methods ----- //
  // calls Residual on the pointer returned from InternalResidual(), while
  // holding a lock
  void Residual(const mjModel* model, const mjData* data, double* residual) const;

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

  virtual void ModifyScene(const mjModel* model, const mjData* data, mjvScene* scene) const {}

  virtual std::string Name() const = 0;
  virtual std::string XmlPath() const = 0;
  virtual std::string URDFPath() const { return {}; }
  virtual std::string GetBaseBodyName() const { return {}; }
  virtual std::vector<std::string> GetEndtipNames() const { /* Ones in URDF, not XML */
    return {};
  }
  virtual std::vector<std::string> GetCollisionLinkNames() const { /* Ones in URDF, not XML */
    return {};
  }
  virtual FabSelfCollisionNamePairs GetSelfCollisionNamePairs() const {
    /* Ones in URDF, not XML */
    return {};
  }
  virtual FabLinkCollisionProps GetCollisionLinkProps() const { return {}; }
  virtual std::vector<FabJointLimit> GetJointLimits() const { return {}; }
  virtual std::vector<FabSubGoalPtr> GetSubGoals() const { return {}; }

  virtual int GetActionDim() const { return 0; }
  virtual int GetTargetObjectId() const { return -1; }
  virtual int GetTargetObjectGeomId() const { return -1; }
  const mjtNum* QueryTargetPos() const { return QueryBodyPos(GetTargetObjectId()); }
  const mjtNum* QueryTargetVel() const { return QueryBodyVel(GetTargetObjectId()); }
  const mjtNum* QueryTargetAcc() const { return QueryBodyAcc(GetTargetObjectId()); }

  virtual bool CheckBlocking(const double start[], const double end[]) { return false; }

  // model
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  Planner* planner_ = nullptr;
  mjvScene* scene_ = nullptr;

  virtual const mjtNum* GetStartPos() { return nullptr; }
  virtual const mjtNum* GetStartVel() { return nullptr; }
  virtual const mjtNum* GetGoalPos() const { return nullptr; }
  virtual const mjtNum* GetGoalVel() const { return nullptr; }
  virtual const mjtNum* GetGoalAcc() const { return nullptr; }

  // NOTE: model_->nq,nv are actuated joints/controls configured in MJ model
  // dof: full dof of the robot
  virtual std::vector<double> QueryJointPos(int dof) const {
    if (model_ && data_) {
      std::vector<double> qpos(dof, 0);
      memcpy(qpos.data(), &data_->qpos[0], std::min(model_->nq, dof) * sizeof(double));
      return qpos;
    }
    return {};
  }

  virtual std::vector<double> QueryJointVel(int dof) const {
    if (model_ && data_) {
      std::vector<double> qvel(dof, 0);
      memcpy(qvel.data(), &data_->qvel[0], std::min(model_->nv, dof) * sizeof(double));
      return qvel;
    }
    return {};
  }

  void SetPlanner(Planner* planner) { planner_ = planner; }
  Planner* Planner() const { return planner_; }

  // Body
  int QueryBodyId(const char* body_name) const {
    return model_ ? mj_name2id(model_, mjOBJ_BODY, body_name) : -1;
  }

  const mjtNum* QueryBodyPos(int body_id) const {
    if (model_) {
      return &data_->xipos[3 * body_id];
    }
    return nullptr;
  }

  mjtNum* QueryBodyVel(int body_id) const {
    if (model_) {
      static double lvel[3] = {0};
#if 1
      // Linear: copy from &cvel[6 * target_id + 3], Angular: copy from &cvel[6 * target_id]
      memcpy(lvel, &data_->cvel[6 * body_id + 3], sizeof(mjtNum) * 3);
#else
      mjtNum vel[6];
      mj_objectVelocity(model_, data_, mjOBJ_BODY, body_id, vel, 0);
      // Linear: copy from &vel[3], Angular: copy from &vel[0]
      memcpy(lvel, &vel[3], sizeof(mjtNum) * 3);
#endif
      return &lvel[0];
    }
    return nullptr;
  }

  mjtNum* QueryBodyAcc(int body_id) const {
    if (model_) {
      static double lacc[3] = {0};
#if 0
      memcpy(lacc, &data_->cacc[6 * body_id + 3], sizeof(mjtNum) * 3);
#else
      mjtNum acc[6];
      mj_objectAcceleration(model_, data_, mjOBJ_BODY, body_id, acc, 0);
      // Linear: copy from &acc[3], Angular: copy from &acc[0]
      memcpy(lacc, &acc[3], sizeof(mjtNum) * 3);
#endif
      return &lacc[0];
    }
    return nullptr;
  }

  // Body mocap
  int GetBodyMocapId(const char* body_name) const {
    if (model_) {
      int body_id = QueryBodyId(body_name);
      return (body_id >= 0) ? model_->body_mocapid[body_id] : -1;
    }
    return -1;
  }

  void SetBodyMocapPos(const char* body_name, const double* pos) const {
    if (data_) {
      int bodyMocapId = GetBodyMocapId(body_name);
      mju_copy3(&data_->mocap_pos[3 * bodyMocapId], pos);
      //  std::cout << body_name << ":" << bodyMocapId << " " << pos[0] << " " << pos[1] << std::endl;
    }
  }

  mjtNum* QueryBodyMocapPos(const char* body_name) const {
    if (data_) {
      int bodyMocapId = GetBodyMocapId(body_name);
      return &data_->mocap_pos[3 * bodyMocapId];
    }
    return nullptr;
  }

  // Geom
  int QueryGeomId(const char* geom_name) const {
    return model_ ? mj_name2id(model_, mjOBJ_GEOM, geom_name) : -1;
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
  std::vector<mjtNum> ray_starts;
  std::vector<mjtNum> ray_ends;
  bool last_goal_reached_ = false;

  // RMP/Fabrics
  using StateX = rmp::State<3>;
  std::vector<StateX> obstacle_statesX_;

  // mutex which should be held on changes to data queried from mjdata
  mutable std::mutex task_data_mutex_;

  // - Actuator
  float actuator_kv = 1.f;

  // - Goal
  virtual bool IsGoalFixed() const { return true; }
  virtual bool QueryGoalReached() { return false; }

  // - Obstacles
  virtual bool AreObstaclesFixed() const { return (GetDynamicObstaclesNum() == 0); }
  int GetObstaclesDim() const { return AreObstaclesFixed() ? 3 : GetDynamicObstaclesDimension(); }
  virtual int GetDynamicObstaclesDimension() const { return 3; }
  virtual void QueryObstacleStatesX() {}

  std::vector<StateX> GetObstacleStatesX() const {
    std::vector<StateX> obstacle_statesX;
    {
      MJPC_LOCK_TASK_DATA_ACCESS;
      obstacle_statesX = obstacle_statesX_;
    }
    return obstacle_statesX;
  }

  int static_obstacles_num = 0;
  int dynamic_obstacles_num = 0;
  virtual int GetStaticObstaclesNum() const { return static_obstacles_num; }
  virtual int GetDynamicObstaclesNum() const { return dynamic_obstacles_num; }
  int GetTotalObstaclesNum() const { return GetStaticObstaclesNum() + GetDynamicObstaclesNum(); }

  // - Constraints
  virtual int GetPlaneConstraintsNum() const { return 0; }

protected:
  // returns a pointer to the ResidualFn instance that's used for physics
  // stepping and plotting, and is internal to the class
  virtual BaseResidualFn* InternalResidual() = 0;
  const BaseResidualFn* InternalResidual() const { return const_cast<Task*>(this)->InternalResidual(); }
  // returns an object which can compute the residual function. the function
  // can assume that a lock on mutex_ is held when it's called
  virtual std::unique_ptr<AbstractResidualFn> ResidualLocked() const = 0;
  // implementation of Task::Transition() which can assume a lock is held.
  // in some cases the transition logic requires calling mj_forward (e.g., for
  // measuring contact forces), which will call the sensor callback, which calls
  // ResidualLocked. In order to avoid such resource contention, mutex_ might be
  // temporarily unlocked, but it must be locked again before returning.
  virtual void TransitionLocked(mjModel* model, mjData* data);

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
