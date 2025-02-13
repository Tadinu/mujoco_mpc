#include "mjpc/planners/gradient/cem_sampler.h"

namespace mjpc {
void CEMSampler::Initialize(mjModel* model, const Task& task) {
  // model
  this->model_ = model;
  action_dim_ = model->nu;

  // task
  this->task_ = &task;

  // sampling noise
  std_initial_ = GetNumberOrDefault(0.1, model,
                                    "sampling_exploration"); // initial variance
  std_min_ = GetNumberOrDefault(0.01, model, "std_min"); // minimum variance
  // fraction of the trajectories that will use full exploration noise
  explore_fraction_ = GetNumberOrDefault(0.0, model, "explore_fraction");

  // set number of trajectories to rollout
  num_trajectory_ = GetNumberOrDefault(10, model, "sampling_trajectories");

  // set number of elite samples max(best 10%, 2)
  n_elite = GetNumberOrDefault(std::max(num_trajectory_ / 10, 2), model, "n_elite");

  if (num_trajectory_ > kMaxTrajectory) {
    mju_error_i("Too many trajectories, %d is the maximum allowed.", kMaxTrajectory);
  }
}

void CEMSampler::Allocate() {
  // policy
  policy_(action_dim_).Allocate(model_, *task_, kMaxTrajectoryHorizon);
  nominal_policy_(action_dim_).Allocate(model_, *task_, kMaxTrajectoryHorizon);
  previous_policy_(action_dim_).Allocate(model_, *task_, kMaxTrajectoryHorizon);

  // scratch
  parameters_scratch.resize(model_->nu * kMaxTrajectoryHorizon);
  // times_scratch.resize(kMaxTrajectoryHorizon);

  // noise
  noise.resize(kMaxTrajectory * (model_->nu * kMaxTrajectoryHorizon));

  // variance
  variance.resize(model_->nu * kMaxTrajectoryHorizon); // (nu * horizon)
}

// reset memory to zeros
void CEMSampler::Reset(int horizon, const double* initial_repeated_action) {
  time = 0.0;

  // policy parameters
  policy_.Reset(horizon, initial_repeated_action);
  nominal_policy_.Reset(horizon, initial_repeated_action);
  previous_policy_.Reset(horizon, initial_repeated_action);

  // scratch
  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);
  // std::fill(times_scratch.begin(), times_scratch.end(), 0.0);

  // noise
  std::fill(noise.begin(), noise.end(), 0.0);

  // variance
  std::fill(variance.begin(), variance.end(), std_initial_ * std_initial_);

  // trajectory samples
  for (int i = 0; i < kMaxTrajectory; i++) {
    candidate_policies_[i].Reset(horizon);
  }

  // improvement
  improvement_ = 0.0;
}

// add random noise to nominal policy
void CEMSampler::AddNoiseToPolicy(GradientPolicy& in_policy, int i) {
  // std
  double std;
  if (i < num_trajectory_ * explore_fraction_) {
    std = std_initial_;
  } else {
    std = std_min_;
  }

  // dimensions
  int num_spline_points = in_policy.num_spline_points;
  int num_parameters = num_spline_points * model_->nu;

  // sampling token
  absl::BitGen gen_;

  // shift index
  int shift = i * (model_->nu * kMaxTrajectoryHorizon);

  // sample noise
  // variance[k] is the standard deviation for the k^th control parameter over
  // the elite samples we draw a bunch of control actions from this distribution
  // (which i indexes) - the noise is stored in `noise`.
  for (int k = 0; k < num_parameters; k++) {
    noise[k + shift] = absl::Gaussian<double>(gen_, 0.0, std::max(std::sqrt(variance[k]), std_min_));
  }

  for (int k = 0; k < in_policy.parameters.size(); k++) {
    // add noise
    mju_addTo(in_policy.parameters.data(), DataAt(noise, shift + k * model_->nu), model_->nu);
    // clamp parameters
    Clamp(in_policy.parameters.data(), model_->actuator_ctrlrange, model_->nu);
  }
}
} // namespace mjpc