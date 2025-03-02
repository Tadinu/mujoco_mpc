// Copyright 2021 DeepMind Technologies Limited
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

#ifndef MJPC_SIMULATE_H_
#define MJPC_SIMULATE_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <ratio>
#include <thread>
#include <vector>

// MuJoCo
#include <mujoco/mujoco.h>
#include <platform_ui_adapter.h>

// mjpc
#include "mjpc/sim_base.h"


#ifdef MJSIMULATE_STATIC
// static library
#define MJSIMULATEAPI
#define MJSIMULATELOCAL
#else
#ifdef MJSIMULATE_DLL_EXPORTS
    #define MJSIMULATEAPI MUJOCO_HELPER_DLL_EXPORT
#else
    #define MJSIMULATEAPI MUJOCO_HELPER_DLL_IMPORT
#endif
  #define MJSIMULATELOCAL MUJOCO_HELPER_DLL_LOCAL
#endif

namespace mjpc {
//-------------------------------- global -----------------------------------------------

// Simulate states not contained in MuJoCo structures
class MJSIMULATEAPI Simulate : public SimulateBase {
public:
  using Clock = std::chrono::steady_clock;
  static_assert(std::ratio_less_equal_v<Clock::period, std::milli>);

  // create object and initialize the simulate ui
  Simulate(std::unique_ptr<mujoco::PlatformUIAdapter> platform_ui_adapter,
           mjvCamera* cam, mjvOption* opt, mjvPerturb* pert,
           std::shared_ptr<Agent> in_agent, bool is_passive) :
    mjpc::SimulateBase(std::move(platform_ui_adapter), cam, opt, pert, std::move(in_agent), is_passive) {
  }

  // Apply UI pose perturbations to model and data
  void ApplyPosePerturbations(int flg_paused);

  // Apply UI force perturbations to model and data
  void ApplyForcePerturbations();

  void Render() override;
};
} // namespace mujoco

#endif
