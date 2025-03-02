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

#include "mjpc/simulate.h"  // mjpc fork

#include <mujoco/mjmodel.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <ratio>
#include <string>

#include "lodepng.h"
#include "mjpc/agent.h"
#include "mjpc/array_safety.h"
#include "mjpc/utilities.h"

// When launched via an App Bundle on macOS, the working directory is the path
// to the App Bundle's resource directory. This causes files to be saved into
// the bundle, which is not the desired behavior. Instead, we open a save dialog
// box to ask the user where to put the file. Since the dialog box logic needs
// to be written in Cost-C, we separate it into a different source file.
#ifdef __APPLE__
std::string GetSavePath(const char* filename);
#else
static std::string GetSavePath(const char* filename) { return filename; }
#endif

namespace mjpc {
namespace mju = ::mujoco::util_mjpc;

//------------------------------------ apply pose perturbations ------------------------------------
void Simulate::ApplyPosePerturbations(int flg_paused) {
  if (this->m_ != nullptr) {
    mjv_applyPerturbPose(this->m_, this->d_, &this->pert, flg_paused); // move mocap bodies only
  }
}

//----------------------------------- apply force perturbations ------------------------------------
void Simulate::ApplyForcePerturbations() {
  if (this->m_ != nullptr) {
    mjv_applyPerturbForce(this->m_, this->d_, &this->pert);
  }
}

//------------------------------------------- rendering --------------------------------------------
// render the ui to the window
void Simulate::Render() {
  // visualization
  if (this->uiloadrequest.load() == 0) {
    // task-specific
    if (this->agent->ActiveTask()->visualize) {
      this->agent->ActiveTask()->ModifyScene(this->m_, this->d_, &this->scn);
    }
    // common to all tasks
    this->agent->ModifyScene(&this->scn);
  }

  // show agent plots
  mjrRect rect = this->uistate.rect[3];
  if (this->agent->plot_enabled && this->uiloadrequest.load() == 0) {
    this->agent->PlotShow(&rect, &this->platform_ui->mjr_context());
  }
  mjpc::SimulateBase::Render();
}
} // namespace mujoco
