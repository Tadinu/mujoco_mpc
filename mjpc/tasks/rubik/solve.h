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

#ifndef MJPC_TASKS_RUBIK_SOLVE_H_
#define MJPC_TASKS_RUBIK_SOLVE_H_

#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
constexpr static double kResetHeight = -0.1;  // cube height to reset

class Rubik : public Task {
 public:
  Rubik() : residual_(this) {
    // path to transition model xml
    std::string path = GetModelPath("rubik/transition_model.xml");

    // load transition model
    constexpr int kErrorLength = 1024;
    char load_error[kErrorLength] = "";
    transition_model_ =
        mj_loadXML(path.c_str(), nullptr, load_error, kErrorLength);
    transition_data_ = mj_makeData(transition_model_);

    // goal cache
    goal_cache_.resize(6 * 10);
    std::fill(goal_cache_.begin(), goal_cache_.end(), 0.0);
  }

  ~Rubik() {
    if (transition_data_) mj_deleteData(transition_data_);
    if (transition_model_) mj_deleteModel(transition_model_);
  }

  inline std::string XmlPath() const override {
    return GetModelPath("rubik/task.xml");
  }
  inline std::string Name() const override { return "Rubik"; }

  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const Rubik* task, int current_mode = 0,
                        int goal_index = 0)
        : BaseResidualFn(task),
          current_mode_(current_mode),
          goal_index_(goal_index) {}
    inline void Residual(const mjModel* model, const mjData* data,
                         double* residual) const override {
      int counter = 0;

      // lock current mode
      int mode = current_mode_;

      // ---------- Residual (0) ----------
      // goal position
      double* goal_position = SensorByName(model, data, "palm_position");

      // system's position
      double* position = SensorByName(model, data, "cube_position");

      // position error
      mju_sub3(residual + counter, position, goal_position);
      counter += 3;

      // ---------- Residual (1) ----------
      // goal orientation
      double* goal_orientation = SensorByName(model, data, "cube_goal_orientation");

      // system's orientation
      double* orientation = SensorByName(model, data, "cube_orientation");
      mju_normalize4(goal_orientation);

      // orientation error
      mju_subQuat(residual + counter, goal_orientation, orientation);
      counter += 3;

      // ---------- Residual (2) ----------
      double* cube_linear_velocity =
          SensorByName(model, data, "cube_linear_velocity");
      mju_copy(residual + counter, cube_linear_velocity, 3);
      counter += 3;

      // ---------- Residual (3) ----------
      mju_copy(residual + counter, data->actuator_force, model->nu);
      counter += model->nu;

      // ---------- Residual (3) ----------
      if (mode == kModeManual || mode == kModeSolve) {
        residual[counter + 0] = data->qpos[11] - parameters_[0];  // red
        residual[counter + 1] = data->qpos[12] - parameters_[1];  // orange
        residual[counter + 2] = data->qpos[13] - parameters_[2];  // blue
        residual[counter + 3] = data->qpos[14] - parameters_[3];  // green
        residual[counter + 4] = data->qpos[15] - parameters_[4];  // white
        residual[counter + 5] = data->qpos[16] - parameters_[5];  // yellow
      } else {
        mju_zero(residual + counter, 6);
      }
      counter += 6;

      // ---------- Residual (4) ----------

      // The unmodified cube model has 20 ball joints: nq=86, nv=66.
      // The patch adds a free joint: nq=93, nv=72.
      // The task adds a ball joint: nq=97, nv=75.
      // The shadow hand has 24 DoFs: nq=121, nv=99.
      // The following two residuals apply for the last 24 entries of qpos and qvel:
      mju_sub(residual + counter, data->qpos + 97, model->key_qpos + 97, 24);
      counter += 24;

      // ---------- Residual (5) ----------
      mju_copy(residual + counter, data->qvel + 75, 24);
      counter += 24;

      // ---------- Residual (6) ----------
      residual[counter++] =
          goal_index_ * 12;  // each face has ~12 cost to unscramble based on
                            // current weights, settings, etc.

      // sensor dim sanity check
      CheckSensorDim(model, counter);
    }

   private:
    friend class Rubik;
    int current_mode_ = 0;
    int goal_index_ = 0;
  };

  inline void TransitionLocked(mjModel* model, mjData* data) override {
    if (transition_model_) {
      if (mode == kModeWait) {
        weight[11] = .01;  // add penalty on joint movement
        // wait
      } else if (mode == kModeScramble) {  // scramble
        double scramble_param = parameters[6];
        int num_scramble = ReinterpretAsInt(scramble_param) + 1;

        // reset
        mju_copy(data->qpos, model->qpos0, model->nq);
        mj_resetData(transition_model_, transition_data_);

        // resize
        face_.resize(num_scramble);
        direction_.resize(num_scramble);
        goal_cache_.resize(6 * num_scramble);

        // set transition model
        for (int i = 0; i < num_scramble; i++) {
          // copy goal face orientations
          mju_copy(goal_cache_.data() + i * 6, transition_data_->qpos, 6);

          // zero out noise
          for (int j = 0; j < 6; j++) {
            double val = goal_cache_[i * 6 + j];
            if (mju_abs(val) < 1.0e-4) {
              goal_cache_[i * 6 + j] = 0.0;
            }
            if (val < 0.5 * mjPI * 1.1 && val > 0.5 * mjPI * 0.9) {
              goal_cache_[i * 6 + j] = 0.5 * mjPI;
            }
            if (val < -0.5 * mjPI * 1.1 && val > -0.5 * mjPI * 0.9) {
              goal_cache_[i * 6 + j] = 0.5 * mjPI;
            }
          }

          // random face + direction
          std::random_device rd;  // Only used once to initialise (seed) engine
          std::mt19937 rng(
              rd());  // Random-number engine used (Mersenne-Twister in this case)

          std::uniform_int_distribution<int> uni_face(0,
                                                      5);  // Guaranteed unbiased
          face_[i] = uni_face(rng);

          std::uniform_int_distribution<int> uni_direction(
              0, 1);  // Guaranteed unbiased
          direction_[i] = uni_direction(rng);
          if (direction_[i] == 0) {
            direction_[i] = -1;
          }

          // set
          for (int t = 0; t < 2000; t++) {
            transition_data_->ctrl[face_[i]] = direction_[i] * 1.57 * t / 2000;
            mj_step(transition_model_, transition_data_);
            mju_copy(data->qpos + 11, transition_data_->qpos, 86);
          }
        }

        // set face goal index
        goal_index_ = num_scramble - 1;
        std::cout << "rotations required: " << num_scramble << "\n";

        // set to solve
        mode = kModeSolve;
        weight[11] = 0;  // remove penalty on joint movement
      } else if (mode == kModeSolve) {  // solve
        // set goal
        mju_copy(parameters.data(), goal_cache_.data() + 6 * goal_index_, 6);

        // check error
        double error[6];
        mju_sub(error, data->qpos + 11, parameters.data(), 6);

        if (mju_norm(error, 6) < 0.085) {
          if (goal_index_ == 0) {
            mode = kModeWait;
            std::cout << "solved!\n";
          } else {
            std::cout << "rotations remaining: " << goal_index_ << "\n";
            goal_index_--;
          }
        }
      }
    }

    // check for drop
    if (data->qpos[6] < kResetHeight) {
      if (mode != kModeWait) { std::cout << "cube fell\n"; }

      // stop optimization
      mode = kModeWait;
    }

    // check goal index
    if (residual_.goal_index_ != goal_index_) {
      residual_.goal_index_ = goal_index_;
    }

    // check for mode change
    if (residual_.current_mode_ != mode) {
      // update mode for residual
      residual_.current_mode_ = mode;
    }
  }


  // modes
  enum RubikMode {
    kModeScramble = 0,
    kModeSolve,
    kModeWait,
    kModeManual,
  };

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.current_mode_,
                                        residual_.goal_index_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  mjModel* transition_model_ = nullptr;
  mjData* transition_data_ = nullptr;
  std::vector<int> face_;
  std::vector<int> direction_;
  std::vector<double> goal_cache_;
  int goal_index_;
};

}  // namespace mjpc

#endif  // MJPC_TASKS_RUBIK_SOLVE_H_
