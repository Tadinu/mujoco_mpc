// Copyright 2023 DeepMind Technologies Limited
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

#ifndef MJPC_ESTIMATORS_KALMAN_H_
#define MJPC_ESTIMATORS_KALMAN_H_

#include <chrono>
#include <string>
#include <vector>
#include <mutex>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/estimators/estimator.h"
#include "mjpc/utilities.h"
#include "mjpc/array_safety.h"

namespace mjpc {
namespace mju = ::mujoco::util_mjpc;
// https://stanford.edu/class/ee363/lectures/kf.pdf
class Kalman : public Estimator {
 public:
  // constructor
  Kalman() = default;
  explicit Kalman(const mjModel* model) {
    Initialize(model);
    Reset();
  }

  // destructor
  ~Kalman() override {
    if (data_) mj_deleteData(data_);
    if (model) mj_deleteModel(model);
  }

  // initialize
  void Initialize(const mjModel* model_) override {
    // model
    if (this->model) mj_deleteModel(this->model);
    this->model = mj_copyModel(nullptr, model_);

    // data
    if (this->data_) mj_deleteData(this->data_);
    data_ = mj_makeData(model_);

    // timestep
    this->model->opt.timestep = GetNumberOrDefault(this->model->opt.timestep,
                                                  model_, "estimator_timestep");

    // dimension
    nstate_ = model_->nq + model_->nv + model_->na;
    ndstate_ = 2 * model_->nv + model_->na;

    // sensor start index
    sensor_start_ = GetNumberOrDefault(0, model_, "estimator_sensor_start");

    // number of sensors
    nsensor_ =
        GetNumberOrDefault(model_->nsensor, model_, "estimator_number_sensor");

    // sensor dimension
    nsensordata_ = 0;
    for (int i = 0; i < nsensor_; i++) {
      nsensordata_ += model_->sensor_dim[sensor_start_ + i];
    }

    // sensor start index
    sensor_start_index_ = 0;
    for (int i = 0; i < sensor_start_; i++) {
      sensor_start_index_ += model_->sensor_dim[i];
    }

    // state
    state.resize(nstate_);

    // covariance
    covariance.resize(ndstate_ * ndstate_);

    // process noise
    noise_process.resize(ndstate_);

    // sensor noise
    noise_sensor.resize(nsensordata_);

    // dynamics Jacobian
    dynamics_jacobian_.resize(ndstate_ * ndstate_);

    // sensor Jacobian
    sensor_jacobian_.resize(model_->nsensordata * ndstate_);

    // sensor error
    sensor_error_.resize(nsensordata_);

    // correction
    correction_.resize(ndstate_);

    // scratch
    tmp0_.resize(ndstate_ * nsensordata_);
    tmp1_.resize(nsensordata_ * nsensordata_);
    tmp2_.resize(nsensordata_ * ndstate_);
    tmp3_.resize(ndstate_ * ndstate_);

    // -- GUI data -- //

    // time step
    gui_timestep_ = this->model->opt.timestep;

    // integrator
    gui_integrator_ = this->model->opt.integrator;

    // process noise
    gui_process_noise_.resize(ndstate_);

    // sensor noise
    gui_sensor_noise_.resize(nsensordata_);
  }

  // reset memory
  void Reset(const mjData* data = nullptr) override {
    // dimension
    int nq = model->nq, nv = model->nv, na = model->na;

    if (data) {
      // state
      mju_copy(state.data(), data->qpos, nq);
      mju_copy(state.data() + nq, data->qvel, nv);
      mju_copy(state.data() + nq + nv, data->act, na);
      time = data->time;
    } else {
      // set home keyframe
      int home_id = mj_name2id(model, mjOBJ_KEY, "home");
      if (home_id >= 0) mj_resetDataKeyframe(model, data_, home_id);

      // state
      mju_copy(state.data(), data_->qpos, nq);
      mju_copy(state.data() + nq, data_->qvel, nv);
      mju_copy(state.data() + nq + nv, data_->act, na);
      time = data_->time;
    }

    // covariance
    mju_eye(covariance.data(), ndstate_);
    double covariance_scl =
        GetNumberOrDefault(1.0e-4, model, "estimator_covariance_initial_scale");
    mju_scl(covariance.data(), covariance.data(), covariance_scl,
            ndstate_ * ndstate_);

    // process noise
    double noise_process_scl =
        GetNumberOrDefault(1.0e-4, model, "estimator_process_noise_scale");
    std::fill(noise_process.begin(), noise_process.end(), noise_process_scl);

    // sensor noise
    double noise_sensor_scl =
        GetNumberOrDefault(1.0e-4, model, "estimator_sensor_noise_scale");
    std::fill(noise_sensor.begin(), noise_sensor.end(), noise_sensor_scl);

    // dynamics Jacobian
    mju_zero(dynamics_jacobian_.data(), ndstate_ * ndstate_);

    // sensor Jacobian
    mju_zero(sensor_jacobian_.data(), model->nsensordata * ndstate_);

    // sensor error
    mju_zero(sensor_error_.data(), nsensordata_);

    // correction
    mju_zero(correction_.data(), ndstate_);

    // timer
    timer_measurement_ = 0.0;
    timer_prediction_ = 0.0;

    // scratch
    std::fill(tmp0_.begin(), tmp0_.end(), 0.0);
    std::fill(tmp1_.begin(), tmp1_.end(), 0.0);
    std::fill(tmp2_.begin(), tmp2_.end(), 0.0);
    std::fill(tmp3_.begin(), tmp3_.end(), 0.0);

    // time step
    gui_timestep_ = model->opt.timestep;

    // integrator
    gui_integrator_ = model->opt.integrator;

    // process noise
    std::fill(gui_process_noise_.begin(), gui_process_noise_.end(), noise_process_scl);

    // sensor noise
    std::fill(gui_sensor_noise_.begin(), gui_sensor_noise_.end(), noise_sensor_scl);
  }

  // update measurement
  void UpdateMeasurement(const double* ctrl, const double* sensor) {
    // start timer
    auto start = std::chrono::steady_clock::now();

    // dimensions
    int nq = model->nq, nv = model->nv, na = model->na, nu = model->nu;

    // set state
    mju_copy(data_->qpos, state.data(), nq);
    mju_copy(data_->qvel, state.data() + nq, nv);
    mju_copy(data_->act, state.data() + nq + nv, na);

    // set ctrl
    mju_copy(data_->ctrl, ctrl, nu);

    // forward to get sensor
    mj_forward(model, data_);

    mju_sub(sensor_error_.data(), sensor + sensor_start_index_,
            data_->sensordata + sensor_start_index_, nsensordata_);

    // -- Kalman gain: P * C' (C * P * C' + R)^-1 -- //

    // sensor Jacobian
    mjd_transitionFD(model, data_, settings.epsilon, settings.flg_centered, NULL,
                    NULL, sensor_jacobian_.data(), NULL);

    // grab rows
    double* C = sensor_jacobian_.data() + sensor_start_index_ * ndstate_;

    // P * C' = tmp0
    mju_mulMatMatT(tmp0_.data(), covariance.data(), C, ndstate_, ndstate_,
                  nsensordata_);

    // C * P * C' = C * tmp0 = tmp1
    mju_mulMatMat(tmp1_.data(), C, tmp0_.data(), nsensordata_, ndstate_,
                  nsensordata_);

    // C * P * C' + R
    for (int i = 0; i < nsensordata_; i++) {
      tmp1_[nsensordata_ * i + i] += noise_sensor[i];
    }

    // factorize: C * P * C' + R
    int rank = mju_cholFactor(tmp1_.data(), nsensordata_, 0.0);
    if (rank < nsensordata_) {
      // TODO(taylor): remove and return status
      mju_error("measurement update rank: (%i / %i)\n", rank, nsensordata_);
    }

    // -- correction: (P * C') * (C * P * C' + R)^-1 * sensor_error -- //

    // tmp2 = (C * P * C' + R) \ sensor_error
    mju_cholSolve(tmp2_.data(), tmp1_.data(), sensor_error_.data(), nsensordata_);

    // correction = (P * C') * (C * P * C' + R) \ sensor_error = tmp0 * tmp2
    mju_mulMatVec(correction_.data(), tmp0_.data(), tmp2_.data(), ndstate_,
                  nsensordata_);

    // -- state update -- //

    // configuration
    mj_integratePos(model, state.data(), correction_.data(), 1.0);

    // velocity + act
    mju_addTo(state.data() + nq, correction_.data() + nv, nv + na);

    // -- covariance update -- //
    // TODO(taylor): Joseph form update ?

    // tmp2 = (C * P * C' + R)^-1 (C * P) = tmp1 \ tmp0'
    for (int i = 0; i < ndstate_; i++) {
      mju_cholSolve(tmp2_.data() + nsensordata_ * i, tmp1_.data(),
                    tmp0_.data() + nsensordata_ * i, nsensordata_);
    }

    // tmp3 = (P * C') * (C * P * C' + R)^-1 (C * P) = tmp0 * tmp2'
    mju_mulMatMatT(tmp3_.data(), tmp0_.data(), tmp2_.data(), ndstate_,
                  nsensordata_, ndstate_);

    // covariance -= tmp3
    mju_subFrom(covariance.data(), tmp3_.data(), ndstate_ * ndstate_);

    // symmetrize
    mju_symmetrize(covariance.data(), covariance.data(), ndstate_);

    // stop timer (ms)
    timer_measurement_ = 1.0e-3 * GetDuration(start);
  }

  // update time
  void UpdatePrediction() {
    // start timer
    auto start = std::chrono::steady_clock::now();

    // dimensions
    int nq = model->nq, nv = model->nv, na = model->na;

    // set state
    mju_copy(data_->qpos, state.data(), nq);
    mju_copy(data_->qvel, state.data() + nq, nv);
    mju_copy(data_->act, state.data() + nq + nv, na);

    // dynamics Jacobian
    mjd_transitionFD(model, data_, settings.epsilon, settings.flg_centered,
                    dynamics_jacobian_.data(), NULL, NULL, NULL);

    // integrate state
    mj_step(model, data_);

    // update state
    mju_copy(state.data(), data_->qpos, nq);
    mju_copy(state.data() + nq, data_->qvel, nv);
    mju_copy(state.data() + nq + nv, data_->act, na);

    // -- update covariance: P = A * P * A' -- //

    //  tmp = P * A'
    mju_mulMatMatT(tmp3_.data(), covariance.data(), dynamics_jacobian_.data(),
                  ndstate_, ndstate_, ndstate_);

    // P = A * tmp
    mju_mulMatMat(covariance.data(), dynamics_jacobian_.data(), tmp3_.data(),
                  ndstate_, ndstate_, ndstate_);

    // process noise
    for (int i = 0; i < ndstate_; i++) {
      covariance[ndstate_ * i + i] += noise_process[i];
    }

    // symmetrize
    mju_symmetrize(covariance.data(), covariance.data(), ndstate_);

    // stop timer
    timer_prediction_ = 1.0e-3 * GetDuration(start);
  }

  // estimator-specific GUI elements
  void GUI(mjUI& ui) override {
    // ----- estimator ------ //
    mjuiDef defEstimator[] = {
        {mjITEM_SECTION, "Estimator", 1, nullptr,
        "AP"},  // needs new section to satisfy mjMAXUIITEM
        {mjITEM_BUTTON, "Reset", 2, nullptr, ""},
        {mjITEM_SLIDERNUM, "Timestep", 2, &gui_timestep_, "1.0e-3 0.1"},
        {mjITEM_SELECT, "Integrator", 2, &gui_integrator_,
        "Euler\nRK4\nImplicit\nImplicitFast"},
        {mjITEM_END}};

    // add estimator
    mjui_add(&ui, defEstimator);

    // -- process noise -- //
    int nv = model->nv;
    int process_noise_shift = 0;
    mjuiDef defProcessNoise[kMaxProcessNoise + 2];

    // separator
    defProcessNoise[0] = {mjITEM_SEPARATOR, "Process Noise Std.", 1};
    process_noise_shift++;

    // add UI elements
    for (int i = 0; i < DimensionProcess(); i++) {
      // element
      defProcessNoise[process_noise_shift] = {
          mjITEM_SLIDERNUM, "", 2, gui_process_noise_.data() + i, "1.0e-8 0.01"};

      // set name
      mju::strcpy_arr(defProcessNoise[process_noise_shift].name, "");

      // shift
      process_noise_shift++;
    }

    // name UI elements
    int jnt_shift = 1;
    std::string jnt_name_pos;
    std::string jnt_name_vel;

    // loop over joints
    for (int i = 0; i < model->njnt; i++) {
      int name_jntadr = model->name_jntadr[i];
      std::string jnt_name(model->names + name_jntadr);

      // get joint type
      int jnt_type = model->jnt_type[i];

      // free
      switch (jnt_type) {
        case mjJNT_FREE:
          // position
          jnt_name_pos = jnt_name + " (pos 0)";
          mju::strcpy_arr(defProcessNoise[jnt_shift + 0].name,
                          jnt_name_pos.c_str());

          jnt_name_pos = jnt_name + " (pos 1)";
          mju::strcpy_arr(defProcessNoise[jnt_shift + 1].name,
                          jnt_name_pos.c_str());

          jnt_name_pos = jnt_name + " (pos 2)";
          mju::strcpy_arr(defProcessNoise[jnt_shift + 2].name,
                          jnt_name_pos.c_str());

          jnt_name_pos = jnt_name + " (pos 3)";
          mju::strcpy_arr(defProcessNoise[jnt_shift + 3].name,
                          jnt_name_pos.c_str());

          jnt_name_pos = jnt_name + " (pos 4)";
          mju::strcpy_arr(defProcessNoise[jnt_shift + 4].name,
                          jnt_name_pos.c_str());

          jnt_name_pos = jnt_name + " (pos 5)";
          mju::strcpy_arr(defProcessNoise[jnt_shift + 5].name,
                          jnt_name_pos.c_str());

          // velocity
          jnt_name_vel = jnt_name + " (vel 0)";
          mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 0].name,
                          jnt_name_vel.c_str());

          jnt_name_vel = jnt_name + " (vel 1)";
          mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 1].name,
                          jnt_name_vel.c_str());

          jnt_name_vel = jnt_name + " (vel 2)";
          mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 2].name,
                          jnt_name_vel.c_str());

          jnt_name_vel = jnt_name + " (vel 3)";
          mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 3].name,
                          jnt_name_vel.c_str());

          jnt_name_vel = jnt_name + " (vel 4)";
          mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 4].name,
                          jnt_name_vel.c_str());

          jnt_name_vel = jnt_name + " (vel 5)";
          mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 5].name,
                          jnt_name_vel.c_str());

          // shift
          jnt_shift += 6;
          break;
        case mjJNT_BALL:
          // position
          jnt_name_pos = jnt_name + " (pos 0)";
          mju::strcpy_arr(defProcessNoise[jnt_shift + 0].name,
                          jnt_name_pos.c_str());

          jnt_name_pos = jnt_name + " (pos 1)";
          mju::strcpy_arr(defProcessNoise[jnt_shift + 1].name,
                          jnt_name_pos.c_str());

          jnt_name_pos = jnt_name + " (pos 2)";
          mju::strcpy_arr(defProcessNoise[jnt_shift + 2].name,
                          jnt_name_pos.c_str());

          // velocity
          jnt_name_vel = jnt_name + " (vel 0)";
          mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 0].name,
                          jnt_name_vel.c_str());

          jnt_name_vel = jnt_name + " (vel 1)";
          mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 1].name,
                          jnt_name_vel.c_str());

          jnt_name_vel = jnt_name + " (vel 2)";
          mju::strcpy_arr(defProcessNoise[nv + jnt_shift + 2].name,
                          jnt_name_vel.c_str());

          // shift
          jnt_shift += 3;
          break;
        case mjJNT_HINGE:
          // position
          jnt_name_pos = jnt_name + " (pos)";
          mju::strcpy_arr(defProcessNoise[jnt_shift].name, jnt_name_pos.c_str());

          // velocity
          jnt_name_vel = jnt_name + " (vel)";
          mju::strcpy_arr(defProcessNoise[nv + jnt_shift].name,
                          jnt_name_vel.c_str());

          // shift
          jnt_shift++;
          break;
        case mjJNT_SLIDE:
          // position
          jnt_name_pos = jnt_name + " (pos)";
          mju::strcpy_arr(defProcessNoise[jnt_shift].name, jnt_name_pos.c_str());

          // velocity
          jnt_name_vel = jnt_name + " (vel)";
          mju::strcpy_arr(defProcessNoise[nv + jnt_shift].name,
                          jnt_name_vel.c_str());

          // shift
          jnt_shift++;
          break;
      }
    }

    // loop over act
    std::string act_str;
    for (int i = 0; i < model->na; i++) {
      act_str = "act (" + std::to_string(i) + ")";
      mju::strcpy_arr(defProcessNoise[nv + jnt_shift + i].name, act_str.c_str());
    }

    // end
    defProcessNoise[process_noise_shift] = {mjITEM_END};

    // add process noise
    mjui_add(&ui, defProcessNoise);

    // -- sensor noise -- //
    int sensor_noise_shift = 0;
    mjuiDef defSensorNoise[kMaxSensorNoise + 2];

    // separator
    defSensorNoise[0] = {mjITEM_SEPARATOR, "Sensor Noise Std.", 1};
    sensor_noise_shift++;

    // loop over sensors
    std::string sensor_str;
    for (int i = 0; i < nsensor_; i++) {
      std::string name_sensor(model->names +
                              model->name_sensoradr[sensor_start_ + i]);
      int dim_sensor = model->sensor_dim[sensor_start_ + i];

      // loop over sensor elements
      for (int j = 0; j < dim_sensor; j++) {
        // element
        defSensorNoise[sensor_noise_shift] = {
            mjITEM_SLIDERNUM, "", 2,
            gui_sensor_noise_.data() + sensor_noise_shift - 1, "1.0e-8 0.01"};

        // sensor name
        sensor_str = name_sensor;

        // add element index
        if (dim_sensor > 1) {
          sensor_str += " (" + std::to_string(j) + ")";
        }

        // set sensor name
        mju::strcpy_arr(defSensorNoise[sensor_noise_shift].name,
                        sensor_str.c_str());

        // shift
        sensor_noise_shift++;
      }
    }

    // end
    defSensorNoise[sensor_noise_shift] = {mjITEM_END};

    // add sensor noise
    mjui_add(&ui, defSensorNoise);
  }

  // set GUI data
  void SetGUIData() override {
    mju_copy(noise_process.data(), gui_process_noise_.data(), DimensionProcess());
    mju_copy(noise_sensor.data(), gui_sensor_noise_.data(), DimensionSensor());
    model->opt.timestep = gui_timestep_;
    model->opt.integrator = gui_integrator_;
  }

  // estimator-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                    int planner_shift, int timer_shift, int planning,
                    int* shift) override {
    // Kalman info
    double estimator_bounds[2] = {-6, 6};

    // covariance trace
    double trace = Trace(covariance.data(), DimensionProcess());
    mjpc::PlotUpdateData(fig_planner, estimator_bounds,
                        fig_planner->linedata[planner_shift + 0][0] + 1,
                        mju_log10(trace), 100, planner_shift + 0, 0, 1, -100);

    // legend
    mju::strcpy_arr(fig_planner->linename[planner_shift + 0], "Covariance Trace");

    // Kalman timers
    double timer_bounds[2] = {0.0, 1.0};

    // measurement update
    PlotUpdateData(fig_timer, timer_bounds,
                  fig_timer->linedata[timer_shift + 0][0] + 1,
                  TimerMeasurement() + TimerPrediction(), 100, timer_shift + 0, 0, 1, -100);

    // legend
    mju::strcpy_arr(fig_timer->linename[timer_shift + 0], "Update");
  }

  // update
  void Update(const double* ctrl, const double* sensor, int mode = 0) override {
    // correct state with latest measurement
    if (mode == 0 || mode == 1) UpdateMeasurement(ctrl, sensor);

    // propagate state forward in time with model
    if (mode == 0 || mode == 2) UpdatePrediction();

    // set time
    time = data_->time;
  }

  // get state
  double* State() override { return state.data(); };

  // get covariance
  double* Covariance() override { return covariance.data(); };

  // get time
  double& Time() override { return time; };

  // get model
  mjModel* Model() override { return model; };

  // get data
  mjData* Data() override { return data_; };

  // get qfrc
  double* Qfrc() override { return data_->qfrc_actuator; };

  // get process noise
  double* ProcessNoise() override { return noise_process.data(); };

  // get sensor noise
  double* SensorNoise() override { return noise_sensor.data(); };

  // dimension process
  int DimensionProcess() const override { return ndstate_; };

  // dimension sensor
  int DimensionSensor() const override { return nsensordata_; };

  // set state
  void SetState(const double* in_state) override {
    mju_copy(this->state.data(), in_state, ndstate_);
  };

  // set time
  void SetTime(double in_time) override { this->time = in_time; }

  // set covariance
  void SetCovariance(const double* in_covariance) override {
    mju_copy(this->covariance.data(), in_covariance, ndstate_ * ndstate_);
  }

  // get measurement timer (ms)
  double TimerMeasurement() const { return timer_measurement_; }

  // get prediction timer (ms)
  double TimerPrediction() const { return timer_prediction_; }

  // model
  mjModel* model = nullptr;

  // state (nstate_)
  std::vector<double> state;
  double time;

  // covariance (ndstate_ x ndstate_)
  std::vector<double> covariance;

  // process noise (ndstate_)
  std::vector<double> noise_process;

  // sensor noise (nsensordata_)
  std::vector<double> noise_sensor;

  // settings
  struct Settings {
    double epsilon = 1.0e-6;
    bool flg_centered = false;
  } settings;

 private:
  // dimensions
  int nstate_;
  int ndstate_;
  int nsensordata_;
  int nsensor_;

  // sensor indexing
  int sensor_start_;
  int sensor_start_index_;

  // data
  mjData* data_ = nullptr;

  // correction (ndstate_)
  std::vector<double> correction_;

  // sensor Jacobian (nsensordata x ndstate_)
  std::vector<double> sensor_jacobian_;

  // dynamics Jacobian (ndstate_ x ndstate_)
  std::vector<double> dynamics_jacobian_;

  // sensor error (nsensordata_)
  std::vector<double> sensor_error_;

  // timer (ms)
  double timer_measurement_;
  double timer_prediction_;

  // scratch
  std::vector<double> tmp0_;
  std::vector<double> tmp1_;
  std::vector<double> tmp2_;
  std::vector<double> tmp3_;

  // -- GUI data -- //

  // time step
  double gui_timestep_;

  // integrator
  int gui_integrator_;

  // process noise
  std::vector<double> gui_process_noise_;

  // sensor noise
  std::vector<double> gui_sensor_noise_;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_KALMAN_H_
