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

#ifndef MJPC_PLANNERS_GRADIENT_POLICY_H_
#define MJPC_PLANNERS_GRADIENT_POLICY_H_

#include <algorithm>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/planners/policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"


namespace mjpc {
using mjpc::spline::SplineInterpolation;

// policy for gradient descent planner
class GradientPolicy : public Policy {
 public:
  // constructor
  GradientPolicy() = default;

  // destructor
  ~GradientPolicy() override = default;

  // ----- methods ----- //
  // allocate memory
  void Allocate(const mjModel* model_, const Task& task_,
                                int horizon) override {
    // model
    this->model = model_;

    // action improvement
    k.resize(model->nu * kMaxTrajectoryHorizon);

    // parameters
    parameters.resize(model->nu * kMaxTrajectoryHorizon);
    parameter_update.resize(model->nu * kMaxTrajectoryHorizon);

    // parameter times
    times.resize(kMaxTrajectoryHorizon);

    // dimensions
    num_parameters = model->nu * kMaxTrajectoryHorizon;

    // spline points
    num_spline_points = GetNumberOrDefault(kMaxTrajectoryHorizon, model,
                                          "gradient_spline_points");

    // representation
    representation = GetNumberOrDefault(SplineInterpolation::kLinearSpline,
                                        model, "gradient_representation");
  }

  // reset memory to zeros
  void Reset(int horizon, const double* initial_repeated_action = nullptr) override {
    std::fill(k.begin(), k.begin() + horizon * model->nu, 0.0);

    // parameters
    if (initial_repeated_action != nullptr) {
      for (int i = 0; i < horizon; ++i) {
        mju_copy(parameters.data() + i * model->nu, initial_repeated_action,
                model->nu);
      }
    } else {
      std::fill(parameters.begin(),
                parameters.begin() + model->nu * horizon, 0.0);
    }
    std::fill(parameter_update.begin(),
              parameter_update.begin() + model->nu * horizon, 0.0);

    // policy parameter times
    std::fill(times.begin(), times.begin() + horizon, 0.0);
  }

  // compute action from policy
  void Action(double* action, const double* state,
                              double time) const override {
    // find times bounds
    int bounds[2];
    FindInterval(bounds, times, time, num_spline_points);

    // ----- get action ----- //

    if (bounds[0] == bounds[1] ||
        representation == SplineInterpolation::kZeroSpline) {
      ZeroInterpolation(action, time, times, parameters.data(), model->nu,
                        num_spline_points);
    } else if (representation == SplineInterpolation::kLinearSpline) {
      LinearInterpolation(action, time, times, parameters.data(), model->nu,
                          num_spline_points);
    } else if (representation == SplineInterpolation::kCubicSpline) {
      CubicInterpolation(action, time, times, parameters.data(), model->nu,
                        num_spline_points);
    }

    // Clamp controls
    Clamp(action, model->actuator_ctrlrange, model->nu);
  }

  // copy policy
  void CopyFrom(const GradientPolicy& policy, int horizon) {
    // action improvement
    mju_copy(k.data(), policy.k.data(), horizon * model->nu);

    // parameters
    mju_copy(parameters.data(), policy.parameters.data(), policy.num_parameters);

    // update
    mju_copy(parameter_update.data(), policy.parameter_update.data(),
            policy.num_parameters);

    // times
    mju_copy(times.data(), policy.times.data(), policy.num_spline_points);

    num_spline_points = policy.num_spline_points;
    num_parameters = policy.num_parameters;
    representation = policy.representation;
  }

  // copy parameters
  void CopyParametersFrom(
      const std::vector<double>& src_parameters,
      const std::vector<double>& src_times) {
    mju_copy(parameters.data(), src_parameters.data(),
            num_spline_points * model->nu);
    mju_copy(times.data(), src_times.data(), num_spline_points);
  }


  // ----- members ----- //
  const mjModel* model;

  std::vector<double> k;  // action improvement

  std::vector<double> parameters;
  std::vector<double> parameter_update;
  std::vector<double> times;
  int num_parameters;
  int num_spline_points;
  mjpc::spline::SplineInterpolation representation;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_GRADIENT_POLICY_H_
