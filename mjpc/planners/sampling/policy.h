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

#ifndef MJPC_PLANNERS_SAMPLING_POLICY_H_
#define MJPC_PLANNERS_SAMPLING_POLICY_H_

#include <mujoco/mujoco.h>
#include <absl/log/check.h>
#include <absl/types/span.h>
#include "mjpc/spline/spline.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"
#include "mjpc/planners/policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/task.h"

namespace mjpc {

using mjpc::spline::TimeSpline;

// policy for sampling planner
class SamplingPolicy : public Policy {
 public:
  // constructor
  SamplingPolicy() = default;

  // destructor
  ~SamplingPolicy() override = default;

  // ----- methods ----- //

  // allocate memory
  void Allocate(const mjModel* model_, const Task& task_,
                int horizon) override {
    // model
    this->model = model_;

    // spline points
    num_spline_points = GetNumberOrDefault(kMaxTrajectoryHorizon, model,
                                          "sampling_spline_points");

    plan = TimeSpline(/*dim=*/model->nu);
    plan.Reserve(num_spline_points);
  }

  // reset memory to zeros
  void Reset(int horizon, const double* initial_repeated_action = nullptr) override {
    plan.Clear();
    if (initial_repeated_action != nullptr) {
      plan.AddNode(0, absl::MakeConstSpan(initial_repeated_action, model->nu));
    }
  }

  // set action from policy
  void Action(double* action_, const double* state_,
              double time_) const override {
    CHECK(action_ != nullptr);
    plan.Sample(time_, absl::MakeSpan(action_, model->nu));

    // Clamp controls
    Clamp(action_, model->actuator_ctrlrange, model->nu);
  }

  // copy policy
  void CopyFrom(const SamplingPolicy& policy, int horizon) {
    this->plan = policy.plan;
    num_spline_points = policy.num_spline_points;
  }

  // copy parameters
  void SetPlan(const TimeSpline& plan_) {
    this->plan = plan_;
  }

  // ----- members ----- //
  const mjModel* model;
  mjpc::spline::TimeSpline plan;
  int num_spline_points;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_SAMPLING_POLICY_H_
