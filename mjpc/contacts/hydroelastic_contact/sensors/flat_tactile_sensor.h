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

#include "mjpc/contacts/hydroelastic_contact/sensors/bvh.h"
#include "mjpc/contacts/hydroelastic_contact/sensors/tactile_sensor_base.h"

namespace mjpc::hydroelastic_contact::sensors {
using namespace mjpc::hydroelastic_contact;

const static float SQRT_2 = 1.41421356237;

struct DynamicFlatTactileConfig {
  float update_rate = 1.0;
  bool window = 1;
  bool visualize = false;
  float resolution = 0.;
  float sampling_resolution = 0.;
  bool use_parallel = false;
  float sigma = false;
};

class FlatTactileSensor : public TactileSensorBase {
public:
  // Overloaded entry point
  bool load(const mjModel *m, mjData *d) override;

  void updateTactileStates(DynamicFlatTactileConfig &config, uint32_t level, const mjModel *m);

protected:
  virtual void internal_update(const mjModel *m, mjData *d,
                               const std::vector<GeomCollisionPtr> &geomCollisions) override;
  void bvh_update(const mjModel *m, mjData *d, const std::vector<GeomCollisionPtr> &geomCollisions);

  /**
   * @brief Renders the tactile sensor tiles in the mujoco scene.
   * @param[in] pressure The pressure values to render.
   * @param[in] rot The rotation of the sensor.
   * @param[in] xpos The centroid position of the sensor in global coordinates.
   * @param[in] topleft The top left corner of the sensor in local coordinates.
   */
  void render_tiles(Eigen::ArrayXXf pressure, mjtNum rot[9], mjtNum xpos[3], mjtNum topleft[3]);

private:
  std::mutex dynamic_param_mutex;
  bool use_parallel = true;
  bool use_gaussian = false;
  bool use_tukey = false;
  bool use_square = false;
  float sigma = -1.0f;
  int sampling_resolution = 5;  // 25 samples per cell
  double resolution;
  int cx = 0, cy = 0;
  // color scaling factors for tactile visualization
  float tactile_running_scale = 3.;
  float tactile_current_scale = 0.;
  Eigen::Vector3d sensor_normal = Eigen::Vector3d::Zero();

  float max_dist = 0.f;
  float di_factor = 0.f;
  float sub_halfwidth = 0.f;
  float rmean = 0.f;
  float rSampling_resolution = 0.f;
};

}  // namespace mjpc::hydroelastic_contact::sensors
