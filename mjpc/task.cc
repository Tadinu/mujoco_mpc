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

#include "mjpc/task.h"

#include <cstring>
#include <mutex>
#include <memory>

#include <absl/strings/match.h>
#include <mujoco/mujoco.h>
#include "mjpc/norm.h"
#include "mjpc/utilities.h"

namespace mjpc {
void Task::UpdateResidual() {
  std::lock_guard<std::mutex> lock(mutex_);
  InternalResidual()->Update();
}

void Task::Transition(mjModel* model, mjData* data) {
  std::lock_guard<std::mutex> lock(mutex_);
  TransitionLocked(model, data);
  InternalResidual()->Update();
}

void Task::CostTerms(double* terms, const double* residual) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return InternalResidual()->CostTerms(terms, residual, /*weighted=*/true);
}

void Task::UnweightedCostTerms(double* terms,
                                         const double* residual) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return InternalResidual()->CostTerms(terms, residual, /*weighted=*/false);
}

double Task::CostValue(const double* residual) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return InternalResidual()->CostValue(residual);
}

}  // namespace mjpc
