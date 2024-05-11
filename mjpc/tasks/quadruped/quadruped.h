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

#ifndef MJPC_TASKS_QUADRUPED_QUADRUPED_H_
#define MJPC_TASKS_QUADRUPED_QUADRUPED_H_

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

// colors of visualisation elements drawn in ModifyScene()
constexpr float kStepRgba[4] = {0.6, 0.8, 0.2, 1};  // step-height cylinders
constexpr float kHullRgba[4] = {0.4, 0.2, 0.8, 1};  // convex hull
constexpr float kAvgRgba[4] = {0.4, 0.2, 0.8, 1};   // average foot position
constexpr float kCapRgba[4] = {0.3, 0.3, 0.8, 1};   // capture point
constexpr float kPcpRgba[4] = {0.5, 0.5, 0.2, 1};   // projected capture point

class QuadrupedFlat : public Task {
 public:
  std::string Name() const override { return "Quadruped Flat"; }
  std::string XmlPath() const override {
    return GetModelPath("quadruped/task_flat.xml");
  }
  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const QuadrupedFlat* task)
        : BaseResidualFn(task) {}
    ResidualFn(const ResidualFn&) = default;
    inline void Residual(const mjModel* model,
                         const mjData* data,
                         double* residual) const override {
      // start counter
      int counter = 0;

      // get foot positions
      double* foot_pos[kNumFoot];
      for (A1Foot foot : kFootAll)
        foot_pos[foot] = data->geom_xpos + 3 * foot_geom_id_[foot];

      // average foot position
      double avg_foot_pos[3];
      AverageFootPos(avg_foot_pos, foot_pos);

      double* torso_xmat = data->xmat + 9*torso_body_id_;
      double* goal_pos = data->mocap_pos + 3*goal_mocap_id_;
      double* compos = SensorByName(model, data, "torso_subtreecom");


      // ---------- Upright ----------
      if (current_mode_ != kModeFlip) {
        if (current_mode_ == kModeBiped) {
          double biped_type = parameters_[biped_type_param_id_];
          int handstand = ReinterpretAsInt(biped_type) ? -1 : 1;
          residual[counter++] = torso_xmat[6] - handstand;
        } else {
          residual[counter++] = torso_xmat[8] - 1;
        }
        residual[counter++] = 0;
        residual[counter++] = 0;
      } else {
        // special handling of flip orientation
        double flip_time = data->time - mode_start_time_;
        double quat[4];
        FlipQuat(quat, flip_time);
        double* torso_xquat = data->xquat + 4*torso_body_id_;
        mju_subQuat(residual + counter, torso_xquat, quat);
        counter += 3;
      }


      // ---------- Height ----------
      // quadrupedal or bipedal height of torso over feet
      double* torso_pos = data->xipos + 3*torso_body_id_;
      bool is_biped = current_mode_ == kModeBiped;
      double height_goal = is_biped ? kHeightBiped : kHeightQuadruped;
      if (current_mode_ == kModeScramble) {
        // disable height term in Scramble
        residual[counter++] = 0;
      } else if (current_mode_ == kModeFlip) {
        // height target for Backflip
        double flip_time = data->time - mode_start_time_;
        residual[counter++] = torso_pos[2] - FlipHeight(flip_time);
      } else {
        residual[counter++] = (torso_pos[2] - avg_foot_pos[2]) - height_goal;
      }


      // ---------- Position ----------
      double* head = data->site_xpos + 3*head_site_id_;
      double target[3];
      if (current_mode_ == kModeWalk) {
        // follow prescribed Walk trajectory
        double mode_time = data->time - mode_start_time_;
        Walk(target, mode_time);
      } else {
        // go to the goal mocap body
        target[0] = goal_pos[0];
        target[1] = goal_pos[1];
        target[2] = goal_pos[2];
      }
      residual[counter++] = head[0] - target[0];
      residual[counter++] = head[1] - target[1];
      residual[counter++] =
          current_mode_ == kModeScramble ? 2 * (head[2] - target[2]) : 0;

      // ---------- Gait ----------
      A1Gait gait = GetGait();
      double step[kNumFoot];
      FootStep(step, GetPhase(data->time), gait);
      for (A1Foot foot : kFootAll) {
        if (is_biped) {
          // ignore "hands" in biped mode
          bool handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
          bool front_hand = !handstand && (foot == kFootFL || foot == kFootFR);
          bool back_hand = handstand && (foot == kFootHL || foot == kFootHR);
          if (front_hand || back_hand) {
            residual[counter++] = 0;
            continue;
          }
        }
        double query[3] = {foot_pos[foot][0], foot_pos[foot][1], foot_pos[foot][2]};

        if (current_mode_ == kModeScramble) {
          double torso_to_goal[3];
          double* goal = data->mocap_pos + 3*goal_mocap_id_;
          mju_sub3(torso_to_goal, goal, torso_pos);
          mju_normalize3(torso_to_goal);
          mju_sub3(torso_to_goal, goal, foot_pos[foot]);
          torso_to_goal[2] = 0;
          mju_normalize3(torso_to_goal);
          mju_addToScl3(query, torso_to_goal, 0.15);
        }

        double ground_height = Ground(model, data, query);
        double height_target = ground_height + kFootRadius + step[foot];
        double height_difference = foot_pos[foot][2] - height_target;
        if (current_mode_ == kModeScramble) {
          // in Scramble, foot higher than target is not penalized
          height_difference = mju_min(0, height_difference);
        }
        residual[counter++] = step[foot] ? height_difference : 0;
      }


      // ---------- Balance ----------
      double* comvel = SensorByName(model, data, "torso_subtreelinvel");
      double capture_point[3];
      double fall_time = mju_sqrt(2*height_goal / 9.81);
      mju_addScl3(capture_point, compos, comvel, fall_time);
      residual[counter++] = capture_point[0] - avg_foot_pos[0];
      residual[counter++] = capture_point[1] - avg_foot_pos[1];


      // ---------- Effort ----------
      mju_scl(residual + counter, data->actuator_force, 2e-2, model->nu);
      counter += model->nu;


      // ---------- Posture ----------
      double* home = KeyQPosByName(model, data, "home");
      mju_sub(residual + counter, data->qpos + 7, home + 7, model->nu);
      if (current_mode_ == kModeFlip) {
        double flip_time = data->time - mode_start_time_;
        if (flip_time < crouch_time_) {
          double* crouch = KeyQPosByName(model, data, "crouch");
          mju_sub(residual + counter, data->qpos + 7, crouch + 7, model->nu);
        } else if (flip_time >= crouch_time_ &&
                  flip_time < jump_time_ + flight_time_) {
          // free legs during flight phase
          mju_zero(residual + counter, model->nu);
        }
      }
      for (A1Foot foot : kFootAll) {
        for (int joint = 0; joint < 3; joint++) {
          residual[counter + 3*foot + joint] *= kJointPostureGain[joint];
        }
      }
      if (current_mode_ == kModeBiped) {
        // loosen the "hands" in Biped mode
        bool handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
        if (handstand) {
          residual[counter + 4] *= 0.03;
          residual[counter + 5] *= 0.03;
          residual[counter + 10] *= 0.03;
          residual[counter + 11] *= 0.03;
        } else {
          residual[counter + 1] *= 0.03;
          residual[counter + 2] *= 0.03;
          residual[counter + 7] *= 0.03;
          residual[counter + 8] *= 0.03;
        }
      }
      counter += model->nu;


      // ---------- Yaw ----------
      double torso_heading[2] = {torso_xmat[0], torso_xmat[3]};
      if (current_mode_ == kModeBiped) {
        int handstand =
            ReinterpretAsInt(parameters_[biped_type_param_id_]) ? 1 : -1;
        torso_heading[0] = handstand * torso_xmat[2];
        torso_heading[1] = handstand * torso_xmat[5];
      }
      mju_normalize(torso_heading, 2);
      double heading_goal = parameters_[ParameterIndex(model, "Heading")];
      residual[counter++] = torso_heading[0] - mju_cos(heading_goal);
      residual[counter++] = torso_heading[1] - mju_sin(heading_goal);


      // ---------- Angular momentum ----------
      mju_copy3(residual + counter, SensorByName(model, data, "torso_angmom"));
      counter +=3;


      // sensor dim sanity check
      CheckSensorDim(model, counter);
    }


   private:
    friend class QuadrupedFlat;
    //  ============  enums  ============
    // modes
    enum A1Mode {
      kModeQuadruped = 0,
      kModeBiped,
      kModeWalk,
      kModeScramble,
      kModeFlip,
      kNumMode
    };

    // feet
    enum A1Foot {
      kFootFL  = 0,
      kFootHL,
      kFootFR,
      kFootHR,
      kNumFoot
    };

    // gaits
    enum A1Gait {
      kGaitStand = 0,
      kGaitWalk,
      kGaitTrot,
      kGaitCanter,
      kGaitGallop,
      kNumGait
    };

    //  ============  constants  ============
    constexpr static A1Foot kFootAll[kNumFoot] = {kFootFL, kFootHL,
                                                  kFootFR, kFootHR};
    constexpr static A1Foot kFootHind[2] = {kFootHL, kFootHR};
    constexpr static A1Gait kGaitAll[kNumGait] = {kGaitStand, kGaitWalk,
                                                  kGaitTrot, kGaitCanter,
                                                  kGaitGallop};

    // gait phase signature (normalized)
    constexpr static double kGaitPhase[kNumGait][kNumFoot] =
    {
    // FL     HL     FR     HR
      {0,     0,     0,     0   },   // stand
      {0,     0.75,  0.5,   0.25},   // walk
      {0,     0.5,   0.5,   0   },   // trot
      {0,     0.33,  0.33,  0.66},   // canter
      {0,     0.4,   0.05,  0.35}    // gallop
    };

    // gait parameters, set when switching into gait
    constexpr static double kGaitParam[kNumGait][6] =
    {
    // duty ratio  cadence  amplitude  balance   upright   height
    // unitless    Hz       meter      unitless  unitless  unitless
      {1,          1,       0,         0,        1,        1},      // stand
      {0.75,       1,       0.03,      0,        1,        1},      // walk
      {0.45,       2,       0.03,      0.2,      1,        1},      // trot
      {0.4,        4,       0.05,      0.03,     0.5,      0.2},    // canter
      {0.3,        3.5,     0.10,      0.03,     0.2,      0.1}     // gallop
    };

    // velocity ranges for automatic gait switching, meter/second
    constexpr static double kGaitAuto[kNumGait] =
    {
      0,     // stand
      0.02,  // walk
      0.02,  // trot
      0.6,   // canter
      2,     // gallop
    };
    // notes:
    // - walk is never triggered by auto-gait
    // - canter actually has a wider range than gallop

    // automatic gait switching: time constant for com speed filter
    constexpr static double kAutoGaitFilter = 0.2;    // second

    // automatic gait switching: minimum time between switches
    constexpr static double kAutoGaitMinTime = 1;     // second

    // target torso height over feet when quadrupedal
    constexpr static double kHeightQuadruped = 0.25;  // meter

    // target torso height over feet when bipedal
    constexpr static double kHeightBiped = 0.6;       // meter

    // radius of foot geoms
    constexpr static double kFootRadius = 0.02;       // meter

    // below this target yaw velocity, walk straight
    constexpr static double kMinAngvel = 0.01;        // radian/second

    // posture gain factors for abduction, hip, knee
    constexpr static double kJointPostureGain[3] = {2, 1, 1};  // unitless

    // flip: crouching height, from which leap is initiated
    constexpr static double kCrouchHeight = 0.15;     // meter

    // flip: leap height, beginning of flight phase
    constexpr static double kLeapHeight = 0.5;        // meter

    // flip: maximum height of flight phase
    constexpr static double kMaxHeight = 0.8;         // meter

    //  ============  methods  ============
    // return internal phase clock
    double GetPhase(double time) const {
      return phase_start_ + (time - phase_start_time_) * phase_velocity_;
    }

    // get gait
    A1Gait GetGait() const {
      if (current_mode_ == kModeBiped)
        return kGaitTrot;
      return static_cast<A1Gait>(ReinterpretAsInt(current_gait_));
    }

    // compute average foot position, depending on mode
    void AverageFootPos(double avg_foot_pos[3],
                        double* foot_pos[kNumFoot]) const {
      if (current_mode_ == kModeBiped) {
        int handstand = ReinterpretAsInt(parameters_[biped_type_param_id_]);
        if (handstand) {
          mju_add3(avg_foot_pos, foot_pos[kFootFL], foot_pos[kFootFR]);
        } else {
          mju_add3(avg_foot_pos, foot_pos[kFootHL], foot_pos[kFootHR]);
        }
        mju_scl3(avg_foot_pos, avg_foot_pos, 0.5);
      } else {
        mju_add3(avg_foot_pos, foot_pos[kFootHL], foot_pos[kFootHR]);
        mju_addTo3(avg_foot_pos, foot_pos[kFootFL]);
        mju_addTo3(avg_foot_pos, foot_pos[kFootFR]);
        mju_scl3(avg_foot_pos, avg_foot_pos, 0.25);
      }
    }

    // return normalized target step height
    double StepHeight(double time, double footphase, double duty_ratio) const {      double angle = fmod(time + mjPI - footphase, 2*mjPI) - mjPI;
      double value = 0;
      if (duty_ratio < 1) {
        angle *= 0.5 / (1 - duty_ratio);
        value = mju_cos(mju_clip(angle, -mjPI/2, mjPI/2));
      }
      return mju_abs(value) < 1e-6 ? 0.0 : value;
    }


    // compute target step height for all feet
    void FootStep(double step[kNumFoot], double time,
                  A1Gait gait) const {
      double amplitude = parameters_[amplitude_param_id_];
      double duty_ratio = parameters_[duty_param_id_];
      for (A1Foot foot : kFootAll) {
        double footphase = 2*mjPI*kGaitPhase[gait][foot];
        step[foot] = amplitude * StepHeight(time, footphase, duty_ratio);
      }
    }

    // walk horizontal position given time
    void Walk(double pos[2], double time) const {
      if (mju_abs(angvel_) < kMinAngvel) {
        // no rotation, go in straight line
        double forward[2] = {heading_[0], heading_[1]};
        mju_normalize(forward, 2);
        pos[0] = position_[0] + heading_[0] + time*speed_*forward[0];
        pos[1] = position_[1] + heading_[1] + time*speed_*forward[1];
      } else {
        // walk on a circle
        double angle = time * angvel_;
        double mat[4] = {mju_cos(angle), -mju_sin(angle),
                        mju_sin(angle),  mju_cos(angle)};
        mju_mulMatVec(pos, mat, heading_, 2, 2);
        pos[0] += position_[0];
        pos[1] += position_[1];
      }
    }

    // height during flip
    double FlipHeight(double time) const {
      if (time >= jump_time_ + flight_time_ + land_time_) {
        return kHeightQuadruped + ground_;
      }
      double h = 0;
      if (time < jump_time_) {
        h = kHeightQuadruped + time * crouch_vel_ + 0.5 * time * time * jump_acc_;
      } else if (time >= jump_time_ && time < jump_time_ + flight_time_) {
        time -= jump_time_;
        h = kLeapHeight + jump_vel_*time - 0.5*9.81*time*time;
      } else if (time >= jump_time_ + flight_time_) {
        time -= jump_time_ + flight_time_;
        h = kLeapHeight - jump_vel_*time + 0.5*land_acc_*time*time;
      }
      return h + ground_;
    }

    // orientation during flip
    void FlipQuat(double quat[4], double time) const {
      //  total rotation = leap + flight + land
      //            2*pi = pi/2 + 5*pi/4 + pi/4
      double angle = 0;
      if (time >= jump_time_ + flight_time_ + land_time_) {
        angle = 2*mjPI;
      } else if (time >= crouch_time_ && time < jump_time_) {
        time -= crouch_time_;
        angle = 0.5 * jump_rot_acc_ * time * time + jump_rot_vel_ * time;
      } else if (time >= jump_time_ && time < jump_time_ + flight_time_) {
        time -= jump_time_;
        angle = mjPI/2 + flight_rot_vel_ * time;
      } else if (time >= jump_time_ + flight_time_) {
        time -= jump_time_ + flight_time_;
        angle = 1.75*mjPI + flight_rot_vel_*time - 0.5*land_rot_acc_ * time * time;
      }
      int flip_dir = ReinterpretAsInt(parameters_[flip_dir_param_id_]);
      double axis[3] = {0, flip_dir ? 1.0 : -1.0, 0};
      mju_axisAngle2Quat(quat, axis, angle);
      mju_mulQuat(quat, orientation_, quat);
    }

    //  ============  task state variables, managed by Transition  ============
    A1Mode current_mode_       = kModeQuadruped;
    double last_transition_time_ = -1;

    // common mode states
    double mode_start_time_  = 0;
    double position_[3]       = {0};

    // walk states
    double heading_[2]        = {0};
    double speed_             = 0;
    double angvel_            = 0;

    // backflip states
    double ground_            = 0;
    double orientation_[4]    = {0};
    double save_gait_switch_  = 0;
    std::vector<double> save_weight_;

    // gait-related states
    double current_gait_      = kGaitStand;
    double phase_start_       = 0;
    double phase_start_time_  = 0;
    double phase_velocity_    = 0;
    double com_vel_[2]        = {0};
    double gait_switch_time_  = 0;

    //  ============  constants, computed in Reset()  ============
    int torso_body_id_        = -1;
    int head_site_id_         = -1;
    int goal_mocap_id_        = -1;
    int gait_param_id_        = -1;
    int gait_switch_param_id_ = -1;
    int flip_dir_param_id_    = -1;
    int biped_type_param_id_  = -1;
    int cadence_param_id_     = -1;
    int amplitude_param_id_   = -1;
    int duty_param_id_        = -1;
    int upright_cost_id_      = -1;
    int balance_cost_id_      = -1;
    int height_cost_id_       = -1;
    int foot_geom_id_[kNumFoot];
    int shoulder_body_id_[kNumFoot];

    // derived kinematic quantities describing flip trajectory
    double gravity_           = 0;
    double jump_vel_          = 0;
    double flight_time_       = 0;
    double jump_acc_          = 0;
    double crouch_time_       = 0;
    double leap_time_         = 0;
    double jump_time_         = 0;
    double crouch_vel_        = 0;
    double land_time_         = 0;
    double land_acc_          = 0;
    double flight_rot_vel_    = 0;
    double jump_rot_vel_      = 0;
    double jump_rot_acc_      = 0;
    double land_rot_acc_      = 0;
  };

  QuadrupedFlat() : residual_(this) {}
  inline void TransitionLocked(mjModel* model, mjData* data) override {
    // ---------- handle mjData reset ----------
    if (data->time < residual_.last_transition_time_ ||
        residual_.last_transition_time_ == -1) {
      if (mode != ResidualFn::kModeQuadruped && mode != ResidualFn::kModeBiped) {
        mode = ResidualFn::kModeQuadruped;  // mode stateful, switch to Quadruped
      }
      residual_.last_transition_time_ = residual_.phase_start_time_ =
          residual_.phase_start_ = data->time;
    }

    // ---------- prevent forbidden mode transitions ----------
    // switching mode, not from quadruped
    if (mode != residual_.current_mode_ &&
        residual_.current_mode_ != ResidualFn::kModeQuadruped) {
      // switch into stateful mode only allowed from Quadruped
      if (mode == ResidualFn::kModeWalk || mode == ResidualFn::kModeFlip) {
        mode = ResidualFn::kModeQuadruped;
      }
    }

    // ---------- handle phase velocity change ----------
    double phase_velocity = 2 * mjPI * parameters[residual_.cadence_param_id_];
    if (phase_velocity != residual_.phase_velocity_) {
      residual_.phase_start_ = residual_.GetPhase(data->time);
      residual_.phase_start_time_ = data->time;
      residual_.phase_velocity_ = phase_velocity;
    }


    // ---------- automatic gait switching ----------
    double* comvel = SensorByName(model, data, "torso_subtreelinvel");
    double beta = mju_exp(-(data->time - residual_.last_transition_time_) /
                          ResidualFn::kAutoGaitFilter);
    residual_.com_vel_[0] = beta * residual_.com_vel_[0] + (1 - beta) * comvel[0];
    residual_.com_vel_[1] = beta * residual_.com_vel_[1] + (1 - beta) * comvel[1];
    // TODO(b/268398978): remove reinterpret, int64_t business
    int auto_switch =
        ReinterpretAsInt(parameters[residual_.gait_switch_param_id_]);
    if (mode == ResidualFn::kModeBiped) {
      // biped always trots
      parameters[residual_.gait_param_id_] =
          ReinterpretAsDouble(ResidualFn::kGaitTrot);
    } else if (auto_switch) {
      double com_speed = mju_norm(residual_.com_vel_, 2);
      for (int64_t gait : ResidualFn::kGaitAll) {
        // scramble requires a non-static gait
        if (mode == ResidualFn::kModeScramble && gait == ResidualFn::kGaitStand)
          continue;
        bool lower = com_speed > ResidualFn::kGaitAuto[gait];
        bool upper = gait == ResidualFn::kGaitGallop ||
                    com_speed <= ResidualFn::kGaitAuto[gait + 1];
        bool wait = mju_abs(residual_.gait_switch_time_ - data->time) >
                    ResidualFn::kAutoGaitMinTime;
        if (lower && upper && wait) {
          parameters[residual_.gait_param_id_] = ReinterpretAsDouble(gait);
          residual_.gait_switch_time_ = data->time;
        }
      }
    }


    // ---------- handle gait switch, manual or auto ----------
    double gait_selection = parameters[residual_.gait_param_id_];
    if (gait_selection != residual_.current_gait_) {
      residual_.current_gait_ = gait_selection;
      ResidualFn::A1Gait gait = residual_.GetGait();
      parameters[residual_.duty_param_id_] = ResidualFn::kGaitParam[gait][0];
      parameters[residual_.cadence_param_id_] = ResidualFn::kGaitParam[gait][1];
      parameters[residual_.amplitude_param_id_] = ResidualFn::kGaitParam[gait][2];
      weight[residual_.balance_cost_id_] = ResidualFn::kGaitParam[gait][3];
      weight[residual_.upright_cost_id_] = ResidualFn::kGaitParam[gait][4];
      weight[residual_.height_cost_id_] = ResidualFn::kGaitParam[gait][5];
    }


    // ---------- Walk ----------
    double* goal_pos = data->mocap_pos + 3*residual_.goal_mocap_id_;
    if (mode == ResidualFn::kModeWalk) {
      double angvel = parameters[ParameterIndex(model, "Walk turn")];
      double speed = parameters[ParameterIndex(model, "Walk speed")];

      // current torso direction
      double* torso_xmat = data->xmat + 9*residual_.torso_body_id_;
      double forward[2] = {torso_xmat[0], torso_xmat[3]};
      mju_normalize(forward, 2);
      double leftward[2] = {-forward[1], forward[0]};

      // switching into Walk or parameters changed, reset task state
      if (mode != residual_.current_mode_ || residual_.angvel_ != angvel ||
          residual_.speed_ != speed) {
        // save time
        residual_.mode_start_time_ = data->time;

        // save current speed and angvel
        residual_.speed_ = speed;
        residual_.angvel_ = angvel;

        // compute and save rotation axis / walk origin
        double axis[2] = {data->xpos[3*residual_.torso_body_id_],
                          data->xpos[3*residual_.torso_body_id_+1]};
        if (mju_abs(angvel) > ResidualFn::kMinAngvel) {
          // don't allow turning with very small angvel
          double d = speed / angvel;
          axis[0] += d * leftward[0];
          axis[1] += d * leftward[1];
        }
        residual_.position_[0] = axis[0];
        residual_.position_[1] = axis[1];

        // save vector from axis to initial goal position
        residual_.heading_[0] = goal_pos[0] - axis[0];
        residual_.heading_[1] = goal_pos[1] - axis[1];
      }

      // move goal
      double time = data->time - residual_.mode_start_time_;
      residual_.Walk(goal_pos, time);
    }


    // ---------- Flip ----------
    double* compos = SensorByName(model, data, "torso_subtreecom");
    if (mode == ResidualFn::kModeFlip) {
      // switching into Flip, reset task state
      if (mode != residual_.current_mode_) {
        // save time
        residual_.mode_start_time_ = data->time;

        // save body orientation, ground height
        mju_copy4(residual_.orientation_,
                  data->xquat + 4 * residual_.torso_body_id_);
        residual_.ground_ = Ground(model, data, compos);

        // save parameters
        residual_.save_weight_ = weight;
        residual_.save_gait_switch_ = parameters[residual_.gait_switch_param_id_];

        // set parameters
        weight[CostTermByName(model, "Upright")] = 0.2;
        weight[CostTermByName(model, "Height")] = 5;
        weight[CostTermByName(model, "Position")] = 0;
        weight[CostTermByName(model, "Gait")] = 0;
        weight[CostTermByName(model, "Balance")] = 0;
        weight[CostTermByName(model, "Effort")] = 0.005;
        weight[CostTermByName(model, "Posture")] = 0.1;
        parameters[residual_.gait_switch_param_id_] = ReinterpretAsDouble(0);
      }

      // time from start of Flip
      double flip_time = data->time - residual_.mode_start_time_;

      if (flip_time >=
          residual_.jump_time_ + residual_.flight_time_ + residual_.land_time_) {
        // Flip ended, back to Quadruped, restore values
        mode = ResidualFn::kModeQuadruped;
        weight = residual_.save_weight_;
        parameters[residual_.gait_switch_param_id_] = residual_.save_gait_switch_;
        goal_pos[0] = data->site_xpos[3*residual_.head_site_id_ + 0];
        goal_pos[1] = data->site_xpos[3*residual_.head_site_id_ + 1];
      }
    }

    // save mode
    residual_.current_mode_ = static_cast<ResidualFn::A1Mode>(mode);
    residual_.last_transition_time_ = data->time;
  }

  // draw task-related geometry in the scene
  inline void ModifyScene(const mjModel* model, const mjData* data,
                          mjvScene* scene) const override {
    // flip target pose
    if (residual_.current_mode_ == ResidualFn::kModeFlip) {
      double flip_time = data->time - residual_.mode_start_time_;
      double* torso_pos = data->xpos + 3*residual_.torso_body_id_;
      double pos[3] = {torso_pos[0], torso_pos[1],
                      residual_.FlipHeight(flip_time)};
      double quat[4];
      residual_.FlipQuat(quat, flip_time);
      double mat[9];
      mju_quat2Mat(mat, quat);
      double size[3] = {0.25, 0.15, 0.05};
      float rgba[4] = {0, 1, 0, 0.5};
      AddGeom(scene, mjGEOM_BOX, size, pos, mat, rgba);

      // don't draw anything else during flip
      return;
    }

    // current foot positions
    double* foot_pos[ResidualFn::kNumFoot];
    for (ResidualFn::A1Foot foot : ResidualFn::kFootAll)
      foot_pos[foot] = data->geom_xpos + 3 * residual_.foot_geom_id_[foot];

    // stance and flight positions
    double flight_pos[ResidualFn::kNumFoot][3];
    double stance_pos[ResidualFn::kNumFoot][3];
    // set to foot horizontal position:
    for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
      flight_pos[foot][0] = stance_pos[foot][0] = foot_pos[foot][0];
      flight_pos[foot][1] = stance_pos[foot][1] = foot_pos[foot][1];
    }

    // ground height below feet
    double ground[ResidualFn::kNumFoot];
    for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
      ground[foot] = Ground(model, data, foot_pos[foot]);
    }

    // step heights
    ResidualFn::A1Gait gait = residual_.GetGait();
    double step[ResidualFn::kNumFoot];
    residual_.FootStep(step, residual_.GetPhase(data->time), gait);

    // draw step height
    for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
      stance_pos[foot][2] = ResidualFn::kFootRadius + ground[foot];
      if (residual_.current_mode_ == ResidualFn::kModeBiped) {
        // skip "hands" in biped mode
        bool handstand =
            ReinterpretAsInt(parameters[residual_.biped_type_param_id_]);
        bool front_hand = !handstand && (foot == ResidualFn::kFootFL ||
                                        foot == ResidualFn::kFootFR);
        bool back_hand = handstand && (foot == ResidualFn::kFootHL ||
                                      foot == ResidualFn::kFootHR);
        if (front_hand || back_hand) continue;
      }
      if (step[foot]) {
        flight_pos[foot][2] = ResidualFn::kFootRadius + step[foot] + ground[foot];
        AddConnector(scene, mjGEOM_CYLINDER, ResidualFn::kFootRadius,
                    stance_pos[foot], flight_pos[foot], kStepRgba);
      }
    }

    // support polygon (currently unused for cost)
    double polygon[2*ResidualFn::kNumFoot];
    for (ResidualFn::A1Foot foot : ResidualFn::kFootAll) {
      polygon[2*foot] = foot_pos[foot][0];
      polygon[2*foot + 1] = foot_pos[foot][1];
    }
    int hull[ResidualFn::kNumFoot];
    int num_hull = Hull2D(hull, ResidualFn::kNumFoot, polygon);
    for (int i=0; i < num_hull; i++) {
      int j = (i + 1) % num_hull;
      AddConnector(scene, mjGEOM_CAPSULE, ResidualFn::kFootRadius/2,
                  stance_pos[hull[i]], stance_pos[hull[j]], kHullRgba);
    }

    // capture point
    bool is_biped = residual_.current_mode_ == ResidualFn::kModeBiped;
    double height_goal =
        is_biped ? ResidualFn::kHeightBiped : ResidualFn::kHeightQuadruped;
    double fall_time = mju_sqrt(2*height_goal / residual_.gravity_);
    double capture[3];
    double* compos = SensorByName(model, data, "torso_subtreecom");
    double* comvel = SensorByName(model, data, "torso_subtreelinvel");
    mju_addScl3(capture, compos, comvel, fall_time);

    // ground under CoM
    double com_ground = Ground(model, data, compos);

    // average foot position
    double feet_pos[3];
    residual_.AverageFootPos(feet_pos, foot_pos);
    feet_pos[2] = com_ground;

    double foot_size[3] = {ResidualFn::kFootRadius, 0, 0};

    // average foot position
    AddGeom(scene, mjGEOM_SPHERE, foot_size, feet_pos, /*mat=*/nullptr, kAvgRgba);

    // capture point
    capture[2] = com_ground;
    AddGeom(scene, mjGEOM_SPHERE, foot_size, capture, /*mat=*/nullptr, kCapRgba);

    // capture point, projected onto hull
    double pcp2[2];
    NearestInHull(pcp2, capture, polygon, hull, num_hull);
    double pcp[3] = {pcp2[0], pcp2[1], com_ground};
    AddGeom(scene, mjGEOM_SPHERE, foot_size, pcp, /*mat=*/nullptr, kPcpRgba);
  }

  //  ============  task-state utilities  ============
  // save task-related ids
  inline void ResetLocked(const mjModel* model) override {
    // ----------  task identifiers  ----------
    residual_.gait_param_id_ = ParameterIndex(model, "select_Gait");
    residual_.gait_switch_param_id_ = ParameterIndex(model, "select_Gait switch");
    residual_.flip_dir_param_id_ = ParameterIndex(model, "select_Flip dir");
    residual_.biped_type_param_id_ = ParameterIndex(model, "select_Biped type");
    residual_.cadence_param_id_ = ParameterIndex(model, "Cadence");
    residual_.amplitude_param_id_ = ParameterIndex(model, "Amplitude");
    residual_.duty_param_id_ = ParameterIndex(model, "Duty ratio");
    residual_.balance_cost_id_ = CostTermByName(model, "Balance");
    residual_.upright_cost_id_ = CostTermByName(model, "Upright");
    residual_.height_cost_id_ = CostTermByName(model, "Height");

    // ----------  model identifiers  ----------
    residual_.torso_body_id_ = mj_name2id(model, mjOBJ_XBODY, "trunk");
    if (residual_.torso_body_id_ < 0) mju_error("body 'trunk' not found");

    residual_.head_site_id_ = mj_name2id(model, mjOBJ_SITE, "head");
    if (residual_.head_site_id_ < 0) mju_error("site 'head' not found");

    int goal_id = mj_name2id(model, mjOBJ_XBODY, "goal");
    if (goal_id < 0) mju_error("body 'goal' not found");

    residual_.goal_mocap_id_ = model->body_mocapid[goal_id];
    if (residual_.goal_mocap_id_ < 0) mju_error("body 'goal' is not mocap");

    // foot geom ids
    int foot_index = 0;
    for (const char* footname : {"FL", "HL", "FR", "HR"}) {
      int foot_id = mj_name2id(model, mjOBJ_GEOM, footname);
      if (foot_id < 0) mju_error_s("geom '%s' not found", footname);
      residual_.foot_geom_id_[foot_index] = foot_id;
      foot_index++;
    }

    // shoulder body ids
    int shoulder_index = 0;
    for (const char* shouldername : {"FL_hip", "HL_hip", "FR_hip", "HR_hip"}) {
      int foot_id = mj_name2id(model, mjOBJ_BODY, shouldername);
      if (foot_id < 0) mju_error_s("body '%s' not found", shouldername);
      residual_.shoulder_body_id_[shoulder_index] = foot_id;
      shoulder_index++;
    }

    // ----------  derived kinematic quantities for Flip  ----------
    residual_.gravity_ = mju_norm3(model->opt.gravity);
    // velocity at takeoff
    residual_.jump_vel_ =
        mju_sqrt(2 * residual_.gravity_ *
                (ResidualFn::kMaxHeight - ResidualFn::kLeapHeight));
    // time in flight phase
    residual_.flight_time_ = 2 * residual_.jump_vel_ / residual_.gravity_;
    // acceleration during jump phase
    residual_.jump_acc_ =
        residual_.jump_vel_ * residual_.jump_vel_ /
        (2 * (ResidualFn::kLeapHeight - ResidualFn::kCrouchHeight));
    // time in crouch sub-phase of jump
    residual_.crouch_time_ =
        mju_sqrt(2 * (ResidualFn::kHeightQuadruped - ResidualFn::kCrouchHeight) /
                residual_.jump_acc_);
    // time in leap sub-phase of jump
    residual_.leap_time_ = residual_.jump_vel_ / residual_.jump_acc_;
    // jump total time
    residual_.jump_time_ = residual_.crouch_time_ + residual_.leap_time_;
    // velocity at beginning of crouch
    residual_.crouch_vel_ = -residual_.jump_acc_ * residual_.crouch_time_;
    // time of landing phase
    residual_.land_time_ =
        2 * (ResidualFn::kLeapHeight - ResidualFn::kHeightQuadruped) /
        residual_.jump_vel_;
    // acceleration during landing
    residual_.land_acc_ = residual_.jump_vel_ / residual_.land_time_;
    // rotational velocity during flight phase (rotates 1.25 pi)
    residual_.flight_rot_vel_ = 1.25 * mjPI / residual_.flight_time_;
    // rotational velocity at start of leap (rotates 0.5 pi)
    residual_.jump_rot_vel_ =
        mjPI / residual_.leap_time_ - residual_.flight_rot_vel_;
    // rotational acceleration during leap (rotates 0.5 pi)
    residual_.jump_rot_acc_ =
        (residual_.flight_rot_vel_ - residual_.jump_rot_vel_) /
        residual_.leap_time_;
    // rotational deceleration during land (rotates 0.25 pi)
    residual_.land_rot_acc_ =
        2 * (residual_.flight_rot_vel_ * residual_.land_time_ - mjPI / 4) /
        (residual_.land_time_ * residual_.land_time_);
  }


 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(residual_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  friend class ResidualFn;
  ResidualFn residual_;
};


class QuadrupedHill : public Task {
 public:
  std::string Name() const override { return "Quadruped Hill"; }
  std::string XmlPath() const override {
    return GetModelPath("quadruped/task_hill.xml");
  }
  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const QuadrupedHill* task, int current_mode = 0)
        : BaseResidualFn(task), current_mode_(current_mode) {}

    // --------------------- Residuals for quadruped task --------------------
    //   Number of residuals: 4
    //     Residual (0): position_z - average(foot position)_z - height_goal
    //     Residual (1): position - goal_position
    //     Residual (2): orientation - goal_orientation
    //     Residual (3): control
    //   Number of parameters: 1
    //     Parameter (1): height_goal
    // -----------------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override {
      // ---------- Residual (0) ----------
      // standing height goal
      double height_goal = parameters_[0];

      // system's standing height
      double standing_height = SensorByName(model, data, "position")[2];

      // average foot height
      double FRz = SensorByName(model, data, "FR")[2];
      double FLz = SensorByName(model, data, "FL")[2];
      double RRz = SensorByName(model, data, "RR")[2];
      double RLz = SensorByName(model, data, "RL")[2];
      double avg_foot_height = 0.25 * (FRz + FLz + RRz + RLz);

      residual[0] = (standing_height - avg_foot_height) - height_goal;

      // ---------- Residual (1) ----------
      // goal position
      const double* goal_position = data->mocap_pos;

      // system's position
      double* position = SensorByName(model, data, "position");

      // position error
      mju_sub3(residual + 1, position, goal_position);

      // ---------- Residual (2) ----------
      // goal orientation
      double goal_rotmat[9];
      const double* goal_orientation = data->mocap_quat;
      mju_quat2Mat(goal_rotmat, goal_orientation);

      // system's orientation
      double body_rotmat[9];
      double* orientation = SensorByName(model, data, "orientation");
      mju_quat2Mat(body_rotmat, orientation);

      mju_sub(residual + 4, body_rotmat, goal_rotmat, 9);

      // ---------- Residual (3) ----------
      mju_copy(residual + 13, data->ctrl, model->nu);
    }

   private:
    friend class QuadrupedHill;
    int current_mode_;
  };
  QuadrupedHill() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override {
      // set mode to GUI selection
      if (mode > 0) {
        residual_.current_mode_ = mode - 1;
      } else {
        // ---------- Compute tolerance ----------
        // goal position
        const double* goal_position = data->mocap_pos;

        // goal orientation
        const double* goal_orientation = data->mocap_quat;

        // system's position
        double* position = SensorByName(model, data, "position");

        // system's orientation
        double* orientation = SensorByName(model, data, "orientation");

        // position error
        double position_error[3];
        mju_sub3(position_error, position, goal_position);
        double position_error_norm = mju_norm3(position_error);

        // orientation error
        double geodesic_distance =
            1.0 - mju_abs(mju_dot(goal_orientation, orientation, 4));

        // ---------- Check tolerance ----------
        double tolerance = 1.5e-1;
        if (position_error_norm <= tolerance && geodesic_distance <= tolerance) {
          // update task state
          residual_.current_mode_ += 1;
          if (residual_.current_mode_ == model->nkey) {
            residual_.current_mode_ = 0;
          }
        }
      }

      // ---------- Set goal ----------
      mju_copy3(data->mocap_pos, model->key_mpos + 3 * residual_.current_mode_);
      mju_copy4(data->mocap_quat, model->key_mquat + 4 * residual_.current_mode_);
    }

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.current_mode_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace mjpc

#endif  // MJPC_TASKS_QUADRUPED_QUADRUPED_H_
