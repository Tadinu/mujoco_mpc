#pragma once

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include <Eigen/Core>
#include <cassert>
#include <chrono>
#include <map>
#include <memory>
#include <shared_mutex>
#include <vector>

// mjpc
#include "mjpc/planners/gradient/policy.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/policy.h"
#include "mjpc/utilities.h"

namespace mjpc {
// Control inputs data augmentation with noise sampling
class CEMSampler {
public:
  // ----- members ----- //
  mjModel* model_;
  const Task* task_;

  // state
  std::vector<double> state_;
  double time;

  // policy
  SamplingPolicy policy_;  // (Guarded by mtx_)
  SamplingPolicy candidate_policies_[kMaxTrajectory];
  SamplingPolicy nominal_policy_;
  SamplingPolicy previous_policy_;

  TrajectoryPtr nominal_trajectory_ = std::make_shared<Trajectory>();

  // scratch
  std::vector<double> parameters_scratch_;

  // number of elite samples
  int n_elite_;

  // improvement
  double improvement_;

  // ----- noise ----- //
  double std_initial_;           // standard deviation for sampling normal: N(0,
                                 // std)
  double std_min_;               // the minimum allowable std
  double explore_fraction_ = 0;  // fraction of trajectories that will use
                                 // std_initial instead of the variance from CEM
  std::vector<double> noise_;
  std::vector<double> variance_;
  int num_trajectory_;
  mutable std::shared_mutex mtx_;

public:
  // initialize data and settings
  void Initialize(mjModel* model, const Task& task);

  // allocate memory
  void Allocate();

  // reset memory to zeros
  void Reset(int horizon, const double* initial_repeated_action = nullptr);

  // optimize nominal policy using random sampling
  void OptimizePolicy(int horizon, ThreadPool& pool);

  // add noise to nominal policy
  void AddNoiseToPolicy(GradientPolicy& in_policy, int i);
};
}  // namespace mjpc