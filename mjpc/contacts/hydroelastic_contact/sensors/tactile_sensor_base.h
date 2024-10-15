/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2023, Bielefeld University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Bielefeld University nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/
/* Authors: Florian Patzelt*/

#pragma once

#include <condition_variable>
#include <vector>

// MuJoCo
#include <mujoco/mujoco.h>

#include "mjpc/contacts/hydroelastic_contact/common_types.h"

namespace mjpc::hydroelastic_contact::sensors {
using namespace mjpc::hydroelastic_contact;

using namespace std::chrono;

struct TactileState {
  std::string name;
  std::vector<float> data;
};

// https://github.com/ubi-agni/tactile_toolbox/blob/melodic-devel/tactile_msgs/msg/TactileState.msg
class TactileSensorBase {
public:
  ~TactileSensorBase() {
    if (vGeoms != nullptr) {
      delete[] vGeoms;
    }
  }
  // Overlead entry point
  virtual bool load(const mjModel *m, mjData *d);
  void update(const mjModel *m, mjData *d, const std::vector<GeomCollisionPtr> &geomCollisions);
  void renderCallback(const mjModel *model, mjData *data, mjvScene *scene);
  void reset();
  std::vector<TactileState> get_tactile_states();

private:
  // Buffer of visual geoms
  // color scaling factors for tactile visualization
  double tactile_running_scale = 3.;
  double tactile_current_scale = 0.;

protected:
  // MuJoCo id of the geom this sensor is attached to
  int geomID;
  // Name of the geom this sensor is attached to
  std::string geomName;

  // update frequency of the sensor
  double updateRate;
  double updatePeriod;

  std::vector<TactileState> tactile_states;
  std::string topicName;
  std::string sensorName;

  bool visualize = false;
  // geom buffer used for visualization
  mjvGeom *vGeoms;
  // number of geoms in vGeoms
  int n_vGeom = 0;
  bool initVGeom(int type, const mjtNum size[3], const mjtNum pos[3], const mjtNum mat[9],
                 const float rgba[4]);
  virtual void internal_update(const mjModel *m, mjData *d,
                               const std::vector<GeomCollisionPtr> &geomCollisions) {};
  std::mutex pause_mutex, state_request_mutex;
  std::condition_variable state_cv;
  bool request_state = false;
  bool paused = false;
  bool setPauseCB();
  int64_t lastUpdate = 0;
};

}  // namespace mjpc::hydroelastic_contact::sensors
