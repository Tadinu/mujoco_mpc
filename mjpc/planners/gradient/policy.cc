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

#include "mjpc/planners/gradient/policy.h"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <vector>

#include "mjpc/spline/spline.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {
using mjpc::spline::SplineInterpolation;

// allocate memory
void GradientPolicy::Allocate(const mjModel* model, const Task& task, int horizon) {
  assert(dim_action > 0);

  // model
  this->model = model;

  // action improvement
  k.resize(dim_action * kMaxTrajectoryHorizon);

  // parameters
  parameters.resize(dim_action * kMaxTrajectoryHorizon);
  parameter_update.resize(dim_action * kMaxTrajectoryHorizon);

  // parameter times
  times.resize(kMaxTrajectoryHorizon);

  // dimensions
  num_parameters = dim_action * kMaxTrajectoryHorizon;

  // spline points
  num_spline_points = GetNumberOrDefault(kMaxTrajectoryHorizon, model, "gradient_spline_points");

  // representation
  representation = GetNumberOrDefault(SplineInterpolation::kLinearSpline, model, "gradient_representation");
}

// reset memory to zeros
void GradientPolicy::Reset(int horizon, const double* initial_repeated_action) {
  std::fill(k.begin(), k.begin() + horizon * dim_action, 0.0);

  // parameters
  if (initial_repeated_action != nullptr) {
    for (int i = 0; i < horizon; ++i) {
      mju_copy(parameters.data() + i * dim_action, initial_repeated_action, dim_action);
    }
  } else {
    std::fill(parameters.begin(), parameters.begin() + dim_action * horizon, 0.0);
  }
  std::fill(parameter_update.begin(), parameter_update.begin() + dim_action * horizon, 0.0);

  // policy parameter times
  std::fill(times.begin(), times.begin() + horizon, 0.0);
}

// compute action from policy
void GradientPolicy::Action(double* action, const double* state, double time,
                            const std::vector<int>& indices) const {
  // find times bounds
  int bounds[2];
  FindInterval(bounds, times, time, num_spline_points);

  // ----- get action ----- //

  if (bounds[0] == bounds[1] || representation == SplineInterpolation::kZeroSpline) {
    ZeroInterpolation(action, time, times, parameters.data(), dim_action, num_spline_points, indices);
  } else if (representation == SplineInterpolation::kLinearSpline) {
    LinearInterpolation(action, time, times, parameters.data(), dim_action, num_spline_points, indices);
  } else if (representation == SplineInterpolation::kCubicSpline) {
    CubicInterpolation(action, time, times, parameters.data(), dim_action, num_spline_points, indices);
  }

  // Clamp controls
  Clamp(action, model->actuator_ctrlrange, dim_action);
}

// copy policy
void GradientPolicy::CopyFrom(const GradientPolicy& policy, int horizon) {
  // action improvement
  mju_copy(k.data(), policy.k.data(), horizon * dim_action);

  // parameters
  mju_copy(parameters.data(), policy.parameters.data(), policy.num_parameters);

  // update
  mju_copy(parameter_update.data(), policy.parameter_update.data(), policy.num_parameters);

  // times
  mju_copy(times.data(), policy.times.data(), policy.num_spline_points);

  num_spline_points = policy.num_spline_points;
  num_parameters = policy.num_parameters;
  representation = policy.representation;
}

// copy parameters
void GradientPolicy::CopyParametersFrom(const std::vector<double>& src_parameters,
                                        const std::vector<double>& src_times) {
  mju_copy(parameters.data(), src_parameters.data(), num_spline_points * dim_action);
  mju_copy(times.data(), src_times.data(), num_spline_points);
}
} // namespace mjpc
