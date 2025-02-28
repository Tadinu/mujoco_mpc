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

#ifndef MJPC_PLANNERS_PLANNER_H_
#define MJPC_PLANNERS_PLANNER_H_

#include <mujoco/mujoco.h>

#include "mjpc/mjcf/mjcf_model.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/urdf_parser/include/model.h"
#include "mjpc/utilities.h"
#include "mjpc/core/mjpc_common.h"

namespace mjpc {
inline constexpr int kMaxTrajectory = 128;
inline constexpr int kMaxTrajectoryLarge = 1028;

// virtual planner
class Planner {
public:
  Planner() = default;

  explicit Planner(const MjOwnerAppType type) : owner_type_(type) {
  }

  // destructor
  virtual ~Planner() = default;

  // initialize data and settings
  virtual void Initialize(mjModel* model, const Task& task) = 0;

  virtual void InitTaskFabrics() {
  }

  // init trajectories
  virtual void InitTrajectory() {
    for (auto& traj : trajectory) {
      traj = std::make_shared<Trajectory>();
    }
  }

  // action
  int GetActionDim() const {
    return action_dim_;
  }

  void SetActionDim(int dim) {
    action_dim_ = dim;
  }

  // action limits
  void SetActionLimits(std::vector<double> limits) {
    action_limits_ = std::move(limits);
  }

  // allocate memory
  virtual void Allocate() = 0;

  // reset memory to zeros
  virtual void Reset(int horizon, const double* initial_repeated_action = nullptr) = 0;

  // set state
  virtual void SetState(const State& state) = 0;

  // optimize nominal policy
  virtual void OptimizePolicy(int horizon, ThreadPool& pool) = 0;

  // compute trajectory using nominal policy
  virtual void NominalTrajectory(int horizon, ThreadPool& pool) = 0;

  // set action from policy
  virtual void ActionFromPolicy(double* action, const double* state, double time,
                                bool use_previous = false) = 0;

  virtual std::vector<double> GetNominalPolicyValues(bool with_noise = false) { return {}; }

  // return trajectory with best total return, or nullptr if no planning
  // iteration has completed
  virtual const Trajectory* BestTrajectory() = 0;

  // visualize planner-specific traces
  virtual void Traces(mjvScene* scn) = 0;

  virtual void ClearTrace() {
  }

  // planner-specific GUI elements
  virtual void GUI(mjUI& ui) = 0;

  // planner-specific plots
  virtual void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift, int timer_shift,
                     int planning, int* shift) = 0;

  // return number of parameters optimized by planner
  virtual int NumParameters() = 0;

  virtual void ResizeMjData(const mjModel* model, int num_threads);

  TrajectoryPtr trajectory[kMaxTrajectory];

  MjOwnerAppType OwnerType() const { return owner_type_; }
  virtual urdf::UrdfModel RobotURDFModel() const { return {}; }
  virtual MjcfModel RobotMJCFModel() const { return {}; }
  bool IsTuningOn() const { return tuning_on_; }
  void SetTuningOn(bool on) { tuning_on_ = on; }

  bool IsPlanningOn() const { return planning_on_; }
  void SetPlanningOn(bool on) { planning_on_ = on; }

  void SetControlCallback(const MjpcPlannerControlCb& cb) {
    control_cb_ = cb;
  }

  std::vector<BaseSolverPtr>& LsqpSolvers() {
    return solvers_;
  }

  const std::vector<UniqueMjData>& RolloutData() const { return data_; }

protected:
  MjOwnerAppType owner_type_ = MjOwnerAppType::MJPC;
  std::vector<UniqueMjData> data_;
  std::vector<BaseSolverPtr> solvers_;
  bool tuning_on_ = false;
  bool planning_on_ = false;
  int action_dim_ = 0;
  std::vector<double> action_limits_; // [2 x action_dim_]
  MjpcPlannerControlCb control_cb_ = nullptr;
};

// additional optional interface for planners that can produce several policy
// proposals
class RankedPlanner : public Planner {
public:
  virtual ~RankedPlanner() = default;
  // optimizes policies, but rather than picking the best, generate up to
  // ncandidates. returns number of candidates created. only called
  // from the planning thread.
  virtual int OptimizePolicyCandidates(int ncandidates, int horizon, ThreadPool& pool) = 0;
  // returns the total return for the nth candidate (or another score to
  // minimize). only called from the planning thread.
  virtual double CandidateScore(int candidate) const = 0;

  // set action from candidate policy. only called from the planning thread.
  virtual void ActionFromCandidatePolicy(double* action, int candidate, const double* state, double time) = 0;

  // sets the nth candidate to the active policy.
  virtual void CopyCandidateToPolicy(int candidate) = 0;
};
} // namespace mjpc

#endif  // MJPC_PLANNERS_PLANNER_H_
