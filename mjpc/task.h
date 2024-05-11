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

#ifndef MJPC_TASK_H_
#define MJPC_TASK_H_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <absl/strings/match.h>
#include <mujoco/mujoco.h>
#include "mjpc/norm.h"
#include "mjpc/utilities.h"

namespace mjpc {

// tolerance for risk-neutral cost
inline constexpr double kRiskNeutralTolerance = 1.0e-6;

// maximum cost terms
inline constexpr int kMaxCostTerms = 128;

class Task;

inline void MissingParameterError(const mjModel* m, int sensorid) {
  mju_error(
      "Cost construction from XML: Missing parameter value."
      " sensor ID = %d (%s)",
      sensorid, m->names + m->name_sensoradr[sensorid]);
}

// abstract class for a residual function
class ResidualFn {
 public:
  virtual ~ResidualFn(){}

  virtual void Residual(const mjModel* model, const mjData* data,
                        double* residual) const{}
  virtual void CostTerms(double* terms, const double* residual,
                         bool weighted) const{}
  virtual double CostValue(const double* residual) const {
    return 0;
  }

  // copies weights and parameters from the Task instance. This should be
  // called from the Task class.
  virtual void Update(){}
};

// base implementation for ResidualFn implementations
class BaseResidualFn : public ResidualFn {
 public:
  explicit BaseResidualFn(const Task* task) : task_(task) {
    Update();
  }
  virtual ~BaseResidualFn(){}

  // compute weighted cost terms
  void CostTerms(double* terms, const double* residual,
                 bool weighted) const override {
    int f_shift = 0;
    int p_shift = 0;
    for (int k = 0; k < num_term_; k++) {
      // running cost
      terms[k] =
          (weighted ? weight_[k] : 1) * Norm(nullptr, nullptr, residual + f_shift,
                                            DataAt(norm_parameter_, p_shift),
                                            dim_norm_residual_[k], norm_[k]);

      // shift residual
      f_shift += dim_norm_residual_[k];

      // shift parameters
      p_shift += num_norm_parameter_[k];
    }
  }

  // compute weighted cost from terms
  double CostValue(const double* residual) const override {
    // cost terms
    double terms[kMaxCostTerms];

    // evaluate
    this->CostTerms(terms, residual, /*weighted=*/true);

    // summation of cost terms
    double cost = 0.0;
    for (int i = 0; i < num_term_; i++) {
      cost += terms[i];
    }

    // exponential risk transformation
    if (mju_abs(risk_) < kRiskNeutralTolerance) {
      return cost;
    } else {
      return (mju_exp(risk_ * cost) - 1.0) / risk_;
    }
  }

  void Update() override {
#if 0
    num_residual_ = task_->num_residual;
    num_term_ = task_->num_term;
    num_trace_ = task_->num_trace;
    dim_norm_residual_ = task_->dim_norm_residual;
    num_norm_parameter_ = task_->num_norm_parameter;
    norm_ = task_->norm;
    weight_ = task_->weight;
    norm_parameter_ = task_->norm_parameter;
    risk_ = task_->risk;
    parameters_ = task_->parameters;
#endif
  }

 protected:
  int num_residual_;
  int num_term_;
  int num_trace_;
  std::vector<int> dim_norm_residual_;
  std::vector<int> num_norm_parameter_;
  std::vector<NormType> norm_;
  std::vector<double> weight_;
  std::vector<double> norm_parameter_;
  double risk_;
  std::vector<double> parameters_;
  const Task* task_;
};

// Thread-safe interface for classes that implement MJPC task specifications
class Task {
 public:
  // constructor
  Task(){}
  virtual ~Task(){}

  // delegates to ResidualLocked, while holding a lock
  inline std::unique_ptr<ResidualFn> Residual() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ResidualLocked();
  }

  // ----- methods ----- //
  // calls Residual on the pointer returned from InternalResidual(), while
  // holding a lock
  inline void Residual(const mjModel* model, const mjData* data,
                                double* residual) const {
    std::lock_guard<std::mutex> lock(mutex_);
    InternalResidual()->Residual(model, data, residual);
  }

  // Must be called whenever parameters or weights change outside Transition or
  // Reset, so that calls to Residual use the new parameters.
  // Calls InternalResidual()->Update() with a lock.
  void UpdateResidual();

  // Changes to data will affect the planner at the next set_state.  Changes to
  // model will only affect the physics and render threads, and will not affect
  // the planner. This is useful for studying planning under model discrepancy,
  // calls TransitionLocked and InternalResidual()->Update() while holding a
  // lock
  void Transition(mjModel* model, mjData* data);

  // get information from model
  // calls ResetLocked and InternalResidual()->Update() while holding a lock
  void Reset(const mjModel* model) {
    std::lock_guard<std::mutex> lock(mutex_);

    // ----- defaults ----- //

    // mode
    mode = 0;

    // risk value
    risk = GetNumberOrDefault(0.0, model, "task_risk");

    // set residual parameters
    this->SetFeatureParameters(model);

    // ----- set costs ----- //
    num_term = 0;
    num_residual = 0;
    num_trace = 0;

    // allocate memory
    dim_norm_residual.resize(kMaxCostTerms);
    num_norm_parameter.resize(kMaxCostTerms);
    norm.resize(kMaxCostTerms);
    weight.resize(kMaxCostTerms);
    norm_parameter.resize(2 * kMaxCostTerms);

    // check user sensor is first
    if (!(model->sensor_type[0] == mjSENS_USER)) {
      mju_error(
          "Cost construction from XML: User sensors specifying residuals must be "
          "specified first and sequentially\n");
    }

    for (int i = 1; true; i++) {
      if (i == model->nsensor || model->sensor_type[i] != mjSENS_USER) {
        num_term = i;
        break;
      }
    }
    if (num_term > kMaxCostTerms) {
      mju_error(
          "Number of cost terms exceeds maximum. Either: 1) reduce number of "
          "terms 2) increase kMaxCostTerms");
    }

    // get number of traces
    for (int i = 0; i < model->nsensor; i++) {
      if (std::strncmp(model->names + model->name_sensoradr[i], "trace",
                      5) == 0) {
        num_trace += 1;
      }
    }
    if (num_trace > kMaxTraces) {
      mju_error("Number of traces should be less than 100\n");
    }

    // loop over sensors
    int parameter_shift = 0;
    for (int i = 0; i < num_term; i++) {
      // residual dimension
      num_residual += model->sensor_dim[i];
      dim_norm_residual[i] = (int)model->sensor_dim[i];

      // user data: [norm, weight, weight_lower, weight_upper, parameters...]
      double* s = model->sensor_user + i * model->nuser_sensor;

      // check number of parameters
      int norm_parameter_dimension = NormParameterDimension(s[0]);
      if (4 + norm_parameter_dimension > model->nuser_sensor) {
        MissingParameterError(model, i);
        return;
      }
      for (int j = 0; j < norm_parameter_dimension; j++) {
        if (s[4 + j] <= 0.0) {
          MissingParameterError(model, i);
          return;
        }
      }
      norm[i] = (NormType)s[0];

      // check Null norm
      if (norm[i] == -1 && dim_norm_residual[i] != 1) {
        MissingParameterError(model, i);
        return;
      }

      weight[i] = s[1];
      num_norm_parameter[i] = norm_parameter_dimension;
      mju_copy(DataAt(norm_parameter, parameter_shift), s + 4,
              num_norm_parameter[i]);
      parameter_shift += num_norm_parameter[i];
    }

    // set residual parameters
    this->SetFeatureParameters(model);

    ResetLocked(model);
    InternalResidual()->Update();
  }

  // calls CostTerms on the pointer returned from InternalResidual(), while
  // holding a lock
  void CostTerms(double* terms, const double* residual) const;

  // calls CostTerms on the pointer returned from InternalResidual(), while
  // holding a lock
  void UnweightedCostTerms(double* terms, const double* residual) const;

  // calls CostValue on the pointer returned from InternalResidual(), while
  // holding a lock
  double CostValue(const double* residual) const;

  virtual void ModifyScene(const mjModel* model, const mjData* data,
                           mjvScene* scene) const {}

  virtual std::string Name() const { return std::string();};
  virtual std::string XmlPath() const {return std::string();};

  // mode
  int mode;

  // GUI toggles
  int reset = 0;
  int visualize = 0;

  // cost parameters
  int num_residual;
  int num_term;
  int num_trace;
  std::vector<int> dim_norm_residual;
  std::vector<int> num_norm_parameter;
  std::vector<NormType> norm;
  std::vector<double> weight;
  std::vector<double> norm_parameter;
  double risk;

  // residual parameters
  std::vector<double> parameters;

 protected:
  // returns a pointer to the ResidualFn instance that's used for physics
  // stepping and plotting, and is internal to the class
  virtual BaseResidualFn* InternalResidual() { return nullptr;}
  const BaseResidualFn* InternalResidual() const {
    return const_cast<Task*>(this)->InternalResidual();
  }
  // returns an object which can compute the residual function. the function
  // can assume that a lock on mutex_ is held when it's called
  virtual std::unique_ptr<ResidualFn> ResidualLocked() const { return nullptr;}
  // implementation of Task::Transition() which can assume a lock is held.
  // in some cases the transition logic requires calling mj_forward (e.g., for
  // measuring contact forces), which will call the sensor callback, which calls
  // ResidualLocked. In order to avoid such resource contention, mutex_ might be
  // temporarily unlocked, but it must be locked again before returning.
  virtual void TransitionLocked(mjModel* model, mjData* data) {}
  // implementation of Task::Reset() which can assume a lock is held
  virtual void ResetLocked(const mjModel* model) {}
  // mutex which should be held on changes to InternalResidual.
  mutable std::mutex mutex_;

 private:
  // initial residual parameters from model
  void SetFeatureParameters(const mjModel* model) {
    // set counter
    int num_parameters = 0;

    // search custom numeric in model for "residual"
    for (int i = 0; i < model->nnumeric; i++) {
      if (absl::StartsWith(model->names + model->name_numericadr[i],
                          "residual_")) {
        num_parameters += 1;
      }
    }

    // allocate memory
    parameters.resize(num_parameters);

    // set values
    int shift = 0;
    for (int i = 0; i < model->nnumeric; i++) {
      if (absl::StartsWith(model->names + model->name_numericadr[i],
                          "residual_select_")) {
        parameters[shift++] = DefaultResidualSelection(model, i);
      } else if (absl::StartsWith(model->names + model->name_numericadr[i],
                                  "residual_")) {
        parameters[shift++] = model->numeric_data[model->numeric_adr[i]];
      }
    }
  }

};

}  // namespace mjpc

#endif  // MJPC_TASK_H_
