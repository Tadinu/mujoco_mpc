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

#include "mjpc/planners/gradient/planner.h"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <chrono>
#include <shared_mutex>

#include "mjpc/array_safety.h"
#include "mjpc/optimizers/stochastic_optimizer.h"
#include "mjpc/planners/cost_derivatives.h"
#include "mjpc/planners/gradient/gradient.h"
#include "mjpc/planners/gradient/policy.h"
#include "mjpc/planners/gradient/settings.h"
#include "mjpc/planners/gradient/spline_mapping.h"
#include "mjpc/planners/model_derivatives.h"
#include "mjpc/planners/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace mju = ::mujoco::util_mjpc;

// initialize planner settings
void GradientPlanner::Initialize(mjModel* model, const Task& task) {
  // delete mjData instances since model might have changed.
  data_.clear();
  // allocate one mjData for nominal.
  ResizeMjData(model, 1);

  // model
  this->model = model;

  // task
  this->task = &task;

  // dimensions
  dim_state = model->nq + model->nv + model->na;     // state dimension
  dim_state_derivative = 2 * model->nv + model->na;  // state derivative dimension
  dim_action = model->nu;                            // action dimension
  dim_sensor = model->nsensordata;                   // number of sensor values
  dim_max = mju_max(mju_max(mju_max(dim_state, dim_state_derivative), dim_action), model->nuser_sensor);
  num_trajectory = GetNumberOrDefault(32, model, "gradient_num_trajectory");

#if MJPC_GRADIENT_PLANNER_USE_CEM
  // cross-entropy
  cross_entropy_sampler.Initialize(model, task);
#endif

  // trajectory_order
  trajectory_order.resize(kMaxTrajectory);
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory_order[i] = i;
  }
}

// allocate memory
void GradientPlanner::Allocate() {
  // state
  state.resize(model->nq + model->nv + model->na);
  mocap.resize(7 * model->nmocap);
  userdata.resize(model->nuserdata);

  // candidate trajectories
  winner = -1;
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i]->Initialize(dim_state, dim_action, task->num_residual, task->num_trace,
                              kMaxTrajectoryHorizon);
    trajectory[i]->Allocate(kMaxTrajectoryHorizon);
  }

  // model derivatives
  model_derivative.Allocate(dim_state_derivative, dim_action, dim_sensor, kMaxTrajectoryHorizon);

  // costs derivatives
  cost_derivative.Allocate(dim_state_derivative, dim_action, task->num_residual, kMaxTrajectoryHorizon,
                           dim_max);

  // gradient descent
  gradient.Allocate(dim_state_derivative, dim_action, kMaxTrajectoryHorizon);

  // spline mapping
  for (auto& mapping : mappings) {
    mapping->Allocate(model->nu);
  }

  // policy
  for (int i = 0; i < kMaxTrajectory; i++) {
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
  }
  policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  previous_policy.Allocate(model, *task, kMaxTrajectoryHorizon);

  // scratch
  parameters_scratch.resize(model->nu * kMaxTrajectoryHorizon);
  times_scratch.resize(kMaxTrajectoryHorizon);

#if MJPC_GRADIENT_PLANNER_USE_CEM
  // cross-entropy
  cross_entropy_sampler.Allocate();
#endif
}

// reset memory to zeros
void GradientPlanner::Reset(int horizon, const double* initial_repeated_action) {
  // state
  std::fill(state.begin(), state.end(), 0.0);
  std::fill(mocap.begin(), mocap.end(), 0.0);
  std::fill(userdata.begin(), userdata.end(), 0.0);
  time = 0.0;

  // model derivatives
  model_derivative.Reset(dim_state_derivative, dim_action, dim_sensor, horizon);

  // cost derivatives
  cost_derivative.Reset(dim_state_derivative, dim_action, task->num_residual, horizon);

  // gradient
  gradient.Reset(dim_state_derivative, dim_action, horizon);

  // policy
  for (int i = 0; i < kMaxTrajectory; i++) {
    candidate_policy[i].Reset(horizon, initial_repeated_action);
  }
  policy.Reset(horizon, initial_repeated_action);
  previous_policy.Reset(horizon, initial_repeated_action);

  // scratch
  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);
  std::fill(times_scratch.begin(), times_scratch.end(), 0.0);

  // candidate trajectories
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i]->Reset(horizon);
  }

  // values
  action_step = 0.0;
  expected = 0.0;
  improvement = 0.0;
  surprise = 0.0;

  // derivative skip
  derivative_skip_ = GetNumberOrDefault(0, model, "derivative_skip");

#if MJPC_GRADIENT_PLANNER_USE_CEM
  // cross-entropy
  cross_entropy_sampler.Reset(horizon, initial_repeated_action);
#endif
}

// set state
void GradientPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(), &this->time);
}

// optimize nominal policy via gradient descent
void GradientPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  ResizeMjData(model, pool.NumThreads());
  // timers
  double nominal_time = 0.0;
  double model_derivative_time = 0.0;
  double cost_derivative_time = 0.0;
  double rollouts_time = 0.0;
  double gradient_time = 0.0;
  double policy_update_time = 0.0;

  // maximum number of trajectories in linesearch
  num_trajectory = mju_min(num_trajectory, kMaxTrajectory);

  // ---- nominal rollout ----- //
  // start timer
  auto nominal_start = std::chrono::steady_clock::now();

  // copy [policy] -> [nominal_policy]
  policy.num_parameters = model->nu * policy.num_spline_points;
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    nominal_policy.CopyFrom(policy, policy.num_spline_points);
  }

  // resample [nominal_policy]: update [parameters_scratch] & write it back to [nominal_policy]
  // update [times_scratch] to curren time
  this->ResamplePolicy(horizon);

  // rollout [nominal_trajectory] by [nominal_policy]
  this->NominalTrajectory(horizon, pool);

  // previous best cost
  double c_prev = nominal_trajectory->total_return;

  // stop timer
  nominal_time = GetDuration(nominal_start);

  // update [nominal_policy], [nominal_trajectory]
  // based on calculation of [model_derivative], [cost_derivative], [gradient]
  double c_best = c_prev;
  int skip = derivative_skip_;
  for (int r = 0; r < settings.max_rollout; r++) {
    // ----- model derivatives ----- //
    // start timer
    auto model_derivative_start = std::chrono::steady_clock::now();

    // compute model and sensor Jacobians
    model_derivative.Compute(model, data_, nominal_trajectory->states.data(),
                             nominal_trajectory->actions.data(), nominal_trajectory->times.data(), dim_state,
                             dim_state_derivative, dim_action, dim_sensor, horizon, settings.fd_tolerance,
                             settings.fd_mode, pool, skip);

    // stop timer
    model_derivative_time += GetDuration(model_derivative_start);

    // -----cost derivatives ----- //
    // start timer
    auto cost_derivative_start = std::chrono::steady_clock::now();

    // compute cost derivatives from [nominal_trajectory] & [model_derivative]
    cost_derivative.Compute(nominal_trajectory->residual.data(), model_derivative.C.data(),
                            model_derivative.D.data(), dim_state_derivative, dim_action, dim_max, dim_sensor,
                            task->num_residual, task->dim_norm_residual.data(), task->num_term,
                            task->weight.data(), task->norm.data(), task->norm_parameter.data(),
                            task->num_norm_parameter.data(), task->risk, horizon, pool);

    // stop timer
    cost_derivative_time += GetDuration(cost_derivative_start);

    // ----- gradient descent ----- //
    // start timer
    auto gradient_start = std::chrono::steady_clock::now();

    // compute action derivatives (gradient's [dV] & [nominal_policy]:[k])
    int gd_status = gradient.Compute(&nominal_policy, &model_derivative, &cost_derivative,
                                     dim_state_derivative, dim_action, horizon);

    // compute spline mapping linear operator from [nominal_policy] & [nominal_trajectory]
    mappings[policy.representation]->Compute(nominal_policy.times, nominal_policy.num_spline_points,
                                             nominal_trajectory->times.data(),
                                             nominal_trajectory->horizon - 1);

    // compute [parameter_update] as total derivatives, from [nominal_policy]:[k]
    mju_mulMatTVec(nominal_policy.parameter_update.data(), mappings[policy.representation]->Get(),
                   nominal_policy.k.data(), model->nu * (nominal_trajectory->horizon - 1),
                   model->nu * nominal_policy.num_spline_points);

    // stop timer
    gradient_time += GetDuration(gradient_start);

    // check for failure
    if (gd_status != 0) return;

    // ----- rollout policy ----- //
    // start timer
    auto rollouts_start = std::chrono::steady_clock::now();

    // improvement step sizes
    LogScale(linesearch_steps, 1.0, settings.min_linesearch_step, num_trajectory - 1);
    linesearch_steps[num_trajectory - 1] = 0.0;

    // rollout all of [trajectory[]] on corresponding [candidate_policy[]] (parallel)
    this->Rollouts(horizon, pool);

    // sort candidate policies and trajectories by score
    for (int i = 0; i < num_trajectory; i++) {
      trajectory_order[i] = i;
    }

    // sort [trajectory_order[]] so that the first ncandidates elements are the best candidates, and
    // the rest are in an unspecified order
    std::partial_sort(trajectory_order.begin(), trajectory_order.begin() + num_trajectory,
                      trajectory_order.begin() + num_trajectory, [&trajectory = trajectory](int a, int b) {
                        return trajectory[a]->total_return < trajectory[b]->total_return;
                      });

#if MJPC_GRADIENT_PLANNER_USE_CEM
    // update policy variance of elite [candidate_policy[]]
    UpdatePolicyVariance();
#endif

    // ----- evaluate rollouts ------ //
    winner = trajectory_order[0];

    // update [nominal_policy] as [winner_policy()], to be used for gradient calculation for the next update
    if (winner != 0) {
      nominal_policy.CopyParametersFrom(winner_policy().parameters, winner_policy().times);
      nominal_trajectory = trajectory[winner];
    }

    // improvement
    action_step = linesearch_steps[winner];
    expected = -action_step * (gradient.dV[0]) - 1.0e-16;
    improvement = c_prev - c_best;
    surprise = mju_min(mju_max(0, improvement / expected), 2);

    // stop timer
    rollouts_time += GetDuration(rollouts_start);
  }  // End settings.maxrollout

  // update nominal policy
  auto policy_update_start = std::chrono::steady_clock::now();

#if MJPC_GRADIENT_PLANNER_USE_CEM
  // copy [parameters_scratch, times_scratch] -> [policy] for [ActionFromPolicy()] & the next OptimizePolicy
  {
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    policy.CopyParametersFrom(cross_entropy_sampler.parameters_scratch_, times_scratch);
  }
#else
  // check for improvement
  if (c_best >= c_prev) {
    winner = num_trajectory - 1;
  }

  // copy [winner_policy()] -> [policy]
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    previous_policy = policy;
    policy.CopyParametersFrom(winner_policy().parameters, winner_policy().times);
  }
#endif

  // stop timer
  policy_update_time += GetDuration(policy_update_start);

  // set timers
  nominal_compute_time = nominal_time;
  model_derivative_compute_time = model_derivative_time;
  cost_derivative_compute_time = cost_derivative_time;
  rollouts_compute_time = rollouts_time;
  gradient_compute_time = gradient_time;
  policy_update_compute_time = policy_update_time;
}

#if MJPC_GRADIENT_PLANNER_USE_CEM
void GradientPlanner::UpdatePolicyVariance() {
  // n_elite_ might change in the GUI - keep constant for in this function
  cross_entropy_sampler.n_elite_ = std::min(cross_entropy_sampler.n_elite_, num_trajectory);
  int n_elite = std::min(cross_entropy_sampler.n_elite_, num_trajectory);

  // dimensions
  int num_spline_points = nominal_policy.num_spline_points;
  int num_parameters = num_spline_points * model->nu;

  // reset [sampling_parameters_scratch]
  auto& sampling_parameters_scratch = cross_entropy_sampler.parameters_scratch_;
  std::fill(sampling_parameters_scratch.begin(), sampling_parameters_scratch.end(), 0.0);

  // update [sampling_parameters_scratch] with [candidate_policy[elite_i]]
  for (int i = 0; i < n_elite; i++) {
    // ordered trajectory index
    int idx = trajectory_order[i];

    // add parameters
    for (int t = 0; t < num_spline_points; t++) {
      for (int j = 0; j < model->nu; j++) {
        sampling_parameters_scratch[t * model->nu + j] += candidate_policy[idx].parameters[j];
      }
    }
  }

  // normalize [sampling_parameters_scratch]
  mju_scl(sampling_parameters_scratch.data(), sampling_parameters_scratch.data(), 1.0 / n_elite,
          num_parameters);

  // compute [variance_] (for noise added to [candidate_policy[]] during rollouts on the next batch)
  // loop over elites (node values of candidate_policy[trajectory_order[0]])
  auto& variance = cross_entropy_sampler.variance_;
  std::fill(variance.begin(), variance.end(), 0.0);  // reset variance to zero
  for (int i = 0; i < n_elite; i++) {
    int idx = trajectory_order[i];
    for (int t = 0; t < num_spline_points; t++) {
      for (int j = 0; j < model->nu; j++) {
        // average
        const double p_avg = parameters_scratch[t * model->nu + j];

        // candidate parameter
        const double pi = candidate_policy[idx].parameters[j];
        const double diff = pi - p_avg;
        variance[t * model->nu + j] += (n_elite >= 1) ? (n_elite * pow(diff, 2)) / (n_elite - 1) : 0;
      }
    }
  }
}
#endif

// compute trajectory using nominal policy
void GradientPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  // nominal policy
  auto frun_nominal_policy = [&cp = nominal_policy](double* action, const double* state, double time) {
    cp.Action(action, state, time);
  };

  // nominal policy rollout
  nominal_trajectory->Rollout(frun_nominal_policy, task, model, data_[0].get(), state.data(), time,
                              mocap.data(), userdata.data(), horizon);
}

// compute action from policy
void GradientPlanner::ActionFromPolicy(double* action, const double* state, double time, bool use_previous) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  if (use_previous) {
    previous_policy.Action(action, state, time);
  } else {
    policy.Action(action, state, time);
  }
}

// update policy for current time
void GradientPlanner::ResamplePolicy(int horizon) {
  // dimensions
  int num_parameters = nominal_policy.num_parameters;
  int num_spline_points = nominal_policy.num_spline_points;

  // time
  double nominal_time = time;
  double time_shift = mju_max((horizon - 1) * model->opt.timestep / (num_spline_points - 1), 1.0e-5);

  // get spline points
  for (int t = 0; t < num_spline_points; t++) {
    times_scratch[t] = nominal_time;
    nominal_policy.Action(DataAt(parameters_scratch, t * model->nu), nullptr, nominal_time);
    nominal_time += time_shift;
  }

  // copy resampled policy parameters
  mju_copy(nominal_policy.parameters.data(), parameters_scratch.data(), num_parameters);
  mju_copy(nominal_policy.times.data(), times_scratch.data(), num_spline_points);

  LinearRange(nominal_policy.times.data(), time_shift, nominal_policy.times[0], num_spline_points);
}

// compute candidate trajectories
void GradientPlanner::Rollouts(int horizon, ThreadPool& pool) {
  // copy [nominal_policy] -> [candidate_policy[i]]
  for (int i = 1; i < num_trajectory; ++i) {
    candidate_policy[i].CopyFrom(nominal_policy, nominal_policy.num_spline_points);
  }

  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectory; i++) {
    pool.Schedule([this, i, &data = data_, &trajectory_i = trajectory[i],
                   &candidate_policy_i = candidate_policy[i], &linesearch_steps_i = linesearch_steps[i],
                   &model = this->model, &task = this->task, &state = this->state, &time = this->time,
                   &mocap = this->mocap, horizon, &userdata = this->userdata]() {
      {
        const std::shared_lock<std::shared_mutex> lock(mtx_);
        // scale improvement: [parameters] += [parameter_update] * [linesearch_steps]
        auto* parameters_i = candidate_policy_i.parameters.data();
        auto* parameters_update_i = candidate_policy_i.parameter_update.data();
        mju_addScl(parameters_i, parameters_i, parameters_update_i, linesearch_steps_i,
                   model->nu * candidate_policy_i.num_spline_points);

#if MJPC_GRADIENT_PLANNER_USE_CEM
        cross_entropy_sampler.AddNoiseToPolicy(candidate_policy_i, i);
#endif
      }

      // policy
      auto frun_feedback_policy = [&candidate_policy_i = candidate_policy_i](
                                      double* action, const double* state, double time) {
        candidate_policy_i.Action(action, state, time);
      };

      // rollout [candidate_policy_i] on [trajectory_i], calculating its [total_return]
      trajectory_i->Rollout(frun_feedback_policy, task, model, data[ThreadPool::WorkerId()].get(),
                            state.data(), time, mocap.data(), userdata.data(), horizon);
    });
  }
  pool.WaitCount(count_before + num_trajectory);
  pool.ResetCount();
}

// return trajectory with best total return
const Trajectory* GradientPlanner::BestTrajectory() {
  return winner >= 0 ? trajectory[winner].get() : nullptr;
}

// visualize candidate traces in GUI
void GradientPlanner::Traces(mjvScene* scn) {
  // sample color
  float color[4];
  color[0] = 1.0;
  color[1] = 1.0;
  color[2] = 1.0;
  color[3] = 1.0;

  // width of a sample trace, in pixels
  double width = GetNumberOrDefault(3, model, "agent_sample_width");

  // scratch
  double zero3[3] = {0};
  double zero9[9] = {0};

  // best
  auto best = this->BestTrajectory();

  for (int k = 0; k < num_trajectory; k++) {
    // plot sample
    for (int i = 0; i < best->horizon - 1; i++) {
      if (scn->ngeom + task->num_trace > scn->maxgeom) break;
      for (int j = 0; j < task->num_trace; j++) {
        // initialize geometry
        mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3, zero9, color);

        // make geometry
        mjv_connector(&scn->geoms[scn->ngeom], mjGEOM_LINE, width,
                      trajectory[k]->trace.data() + 3 * task->num_trace * i + 3 * j,
                      trajectory[k]->trace.data() + 3 * task->num_trace * (i + 1) + 3 * j);

        // increment number of geometries
        scn->ngeom += 1;
      }
    }
  }
}

// planner-specific GUI elements
void GradientPlanner::GUI(mjUI& ui) {
  mjuiDef defGradientPlanner[] = {{mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory, "0 1"},
                                  // {mjITEM_RADIO, "Action Lmt.", 2, &settings.action_limits, "Off\nOn"},
                                  // {mjITEM_SLIDERINT, "Iterations", 2, &settings.max_rollout, "1 128"},
                                  {mjITEM_SELECT, "Spline", 2, &policy.representation, "Zero\nLinear\nCubic"},
                                  {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
                                  {mjITEM_SLIDERINT, "Deriv. Skip", 2, &derivative_skip_, "0 16"},
                                  {mjITEM_END}};

  // set number of trajectory slider limits
  mju::sprintf_arr(defGradientPlanner[0].other, "%i %i", 1, kMaxTrajectory);

  // set spline point limits
  mju::sprintf_arr(defGradientPlanner[2].other, "%i %i", kMinGradientSplinePoints, kMaxGradientSplinePoints);

  // add gradient descent planner
  mjui_add(&ui, defGradientPlanner);
}

// planner-specific plots
void GradientPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift, int timer_shift,
                            int planning, int* shift) {
  // bounds
  double planner_bounds[2] = {-6, 6};

  // ----- planner ----- //
  // step size
  mjpc::PlotUpdateData(fig_planner, planner_bounds, fig_planner->linedata[0 + planner_shift][0] + 1,
                       mju_log10(mju_max(action_step, 1.0e-6)), 100, 0, 0, 1, -100);

  // // improvement
  // mjpc::PlotUpdateData(
  //     fig_planner, planner_bounds, fig_planner->linedata[1 +
  //     planner_shift][0] + 1, mju_log10(mju_max(improvement, 1.0e-6)), 100, 1
  //     + planner_shift, 0, 1, -100);

  // // expected
  // mjpc::PlotUpdateData(
  //     fig_planner, planner_bounds, fig_planner->linedata[2 +
  //     planner_shift][0] + 1, mju_log10(mju_max(expected, 1.0e-6)), 100, 2 +
  //     planner_shift, 0, 1, -100);

  // // surprise
  // mjpc::PlotUpdateData(
  //     fig_planner, planner_bounds, fig_planner->linedata[3 +
  //     planner_shift][0] + 1, mju_log10(mju_max(surprise, 1.0e-6)), 100, 3 +
  //     planner_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Step Size");
  // mju::strcpy_arr(fig_planner->linename[1 + planner_shift], "Improvement");
  // mju::strcpy_arr(fig_planner->linename[2 + planner_shift], "Expected");
  // mju::strcpy_arr(fig_planner->linename[3 + planner_shift], "Surprise");

  // ranges
  fig_planner->range[1][0] = planner_bounds[0];
  fig_planner->range[1][1] = planner_bounds[1];

  // ----- timer ----- //
  double timer_bounds[2] = {0.0, 1.0};

  // update plots
  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[0 + timer_shift][0] + 1,
                 1.0e-3 * nominal_compute_time * planning, 100, 0 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[1 + timer_shift][0] + 1,
                 1.0e-3 * model_derivative_compute_time * planning, 100, 1 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[2 + timer_shift][0] + 1,
                 1.0e-3 * cost_derivative_compute_time * planning, 100, 2 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[3 + timer_shift][0] + 1,
                 1.0e-3 * gradient_compute_time * planning, 100, 4, 3 + timer_shift, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[4 + timer_shift][0] + 1,
                 1.0e-3 * rollouts_compute_time * planning, 100, 4 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[5 + timer_shift][0] + 1,
                 1.0e-3 * policy_update_compute_time * planning, 100, 5 + timer_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Nominal");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Model Deriv.");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Cost Deriv.");
  mju::strcpy_arr(fig_timer->linename[3 + timer_shift], "Gradient");
  mju::strcpy_arr(fig_timer->linename[4 + timer_shift], "Rollouts");
  mju::strcpy_arr(fig_timer->linename[5 + timer_shift], "Policy Update");

  // planner shift
  shift[0] += 1;

  // timer shift
  shift[1] += 6;
}

}  // namespace mjpc
