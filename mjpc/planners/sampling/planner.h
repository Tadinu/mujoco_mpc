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

#ifndef MJPC_PLANNERS_SAMPLING_PLANNER_H_
#define MJPC_PLANNERS_SAMPLING_PLANNER_H_

#include <mujoco/mujoco.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <shared_mutex>
#include <vector>
#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/array_safety.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace mju = ::mujoco::util_mjpc;
using mjpc::spline::SplineInterpolation;
using mjpc::spline::TimeSpline;

// sampling planner limits
inline constexpr int MinSamplingSplinePoints = 1;
inline constexpr int MaxSamplingSplinePoints = 36;
inline constexpr double MinNoiseStdDev = 0.0;
inline constexpr double MaxNoiseStdDev = 1.0;

class SamplingPlanner : public RankedPlanner {
 public:
  // constructor
  SamplingPlanner() = default;

  // destructor
  ~SamplingPlanner() override = default;

  // ----- methods ----- //
  // initialize data and settings
  void Initialize(mjModel* model_, const Task& task_) override {
    // delete mjData instances since model might have changed.
    data_.clear();
    // allocate one mjData for nominal.
    ResizeMjData(model_, 1);

    // model
    this->model = model_;

    // task
    this->task = &task_;

    // sampling noise std
    noise_exploration[0] = GetNumberOrDefault(0.1, model, "sampling_exploration");

    // optional second std (defaults to 0)
    int se_id = mj_name2id(model, mjOBJ_NUMERIC, "sampling_exploration");
    if (se_id >= 0 && model->numeric_size[se_id] > 1) {
      int se_adr = model->numeric_adr[se_id];
      noise_exploration[1] = model->numeric_data[se_adr+1];
    }

    // set number of trajectories to rollout
    num_trajectory_ = GetNumberOrDefault(10, model, "sampling_trajectories");

    interpolation_ = GetNumberOrDefault(SplineInterpolation::kCubicSpline, model,
                                        "sampling_representation");
    sliding_plan_ = GetNumberOrDefault(0, model, "sampling_sliding_plan");

    if (num_trajectory_ > kMaxTrajectory) {
      mju_error_i("Too many trajectories, %d is the maximum allowed.",
                  kMaxTrajectory);
    }

    winner = 0;
  }

  // allocate memory
  void Allocate() override {
    // initial state
    int num_state = model->nq + model->nv + model->na;

    // state
    state.resize(num_state);
    mocap.resize(7 * model->nmocap);
    userdata.resize(model->nuserdata);

    // policy
    policy.Allocate(model, *task, kMaxTrajectoryHorizon);
    previous_policy.Allocate(model, *task, kMaxTrajectoryHorizon);
    plan_scratch = TimeSpline(/*dim=*/model->nu);

    // noise
    noise.resize(kMaxTrajectory * (model->nu * kMaxTrajectoryHorizon));

    // trajectory and parameters
    winner = -1;
    for (int i = 0; i < kMaxTrajectory; i++) {
      trajectory[i].Initialize(num_state, model->nu, task->num_residual,
                              task->num_trace, kMaxTrajectoryHorizon);
      trajectory[i].Allocate(kMaxTrajectoryHorizon);
      candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
    }
  }

  // reset memory to zeros
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr) override {
    // state
    std::fill(state.begin(), state.end(), 0.0);
    std::fill(mocap.begin(), mocap.end(), 0.0);
    std::fill(userdata.begin(), userdata.end(), 0.0);
    time = 0.0;

    // policy parameters
    policy.Reset(horizon, initial_repeated_action);
    previous_policy.Reset(horizon, initial_repeated_action);

    // scratch
    plan_scratch.Clear();

    // noise
    std::fill(noise.begin(), noise.end(), 0.0);

    // trajectory samples
    for (int i = 0; i < kMaxTrajectory; i++) {
      trajectory[i].Reset(kMaxTrajectoryHorizon);
      candidate_policy[i].Reset(horizon, initial_repeated_action);
    }

    for (const auto& d : data_) {
      if (initial_repeated_action) {
        mju_copy(d->ctrl, initial_repeated_action, model->nu);
      } else {
        mju_zero(d->ctrl, model->nu);
      }
    }

    // improvement
    improvement = 0.0;

    // winner
    winner = 0;
  }

  // set state
  void SetState(const State& state_) override {
    state_.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
                  &this->time);
  }

  int OptimizePolicyCandidates(int ncandidates, int horizon,
                              ThreadPool& pool) override {
    // if num_trajectory_ has changed, use it in this new iteration.
    // num_trajectory_ might change while this function runs. Keep it constant
    // for the duration of this function.
    int num_trajectory = num_trajectory_;
    ncandidates = std::min(ncandidates, num_trajectory);
    ResizeMjData(model, pool.NumThreads());

    // ----- rollout noisy policies ----- //
    // start timer
    auto rollouts_start = std::chrono::steady_clock::now();

    // simulate noisy policies
    policy.plan.SetInterpolation(interpolation_);
    this->Rollouts(num_trajectory, horizon, pool);

    // sort candidate policies and trajectories by score
    trajectory_order.clear();
    trajectory_order.reserve(num_trajectory);
    for (int i = 0; i < num_trajectory; i++) {
      trajectory_order.push_back(i);
    }

    // sort so that the first ncandidates elements are the best candidates, and
    // the rest are in an unspecified order
    std::partial_sort(
        trajectory_order.begin(), trajectory_order.begin() + ncandidates,
        trajectory_order.end(), [trajectory = trajectory](int a, int b) {
          return trajectory[a].total_return < trajectory[b].total_return;
        });

    // stop timer
    rollouts_compute_time = GetDuration(rollouts_start);

    return ncandidates;
  }

  // optimize nominal policy using random sampling
  void OptimizePolicy(int horizon, ThreadPool& pool) override {
    // resample nominal policy to current time
    this->UpdateNominalPolicy(horizon);

    OptimizePolicyCandidates(1, horizon, pool);

    // ----- update policy ----- //
    // start timer
    auto policy_update_start = std::chrono::steady_clock::now();

    CopyCandidateToPolicy(0);

    // improvement: compare nominal to winner
    double best_return = trajectory[0].total_return;
    improvement = mju_max(best_return - trajectory[winner].total_return, 0.0);

    // stop timer
    policy_update_compute_time = GetDuration(policy_update_start);
  }

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override {
    // set policy
    auto nominal_policy = [&cp = candidate_policy[0]](
                           double* action_, const double* state_, double time_) {
      cp.Action(action_, state_, time_);
    };

    // rollout nominal policy
    trajectory[0].Rollout(nominal_policy, task, model, data_[0].get(),
                          state.data(), time, mocap.data(), userdata.data(),
                          horizon);
  }

  // set action from policy
  void ActionFromPolicy(double* action_, const double* state_,
                        double time_, bool use_previous_ = false) override {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    if (use_previous_) {
      previous_policy.Action(action_, state_, time_);
    } else {
      policy.Action(action_, state_, time_);
    }
  }

  // update policy via resampling
  void UpdateNominalPolicy(int horizon) {
    // dimensions
    int num_spline_points = candidate_policy[winner].num_spline_points;

    // set time
    double nominal_time = time;
    double time_horizon = (horizon - 1) * model->opt.timestep;

    if (sliding_plan_) {
      // extra points required outside of the horizon window
      int extra_points;
      switch (interpolation_) {
        case spline::SplineInterpolation::kZeroSpline:
          extra_points = 1;
          break;
        case spline::SplineInterpolation::kLinearSpline:
          extra_points = 2;
          break;
        case spline::SplineInterpolation::kCubicSpline:
          extra_points = 4;
          break;
      }

      // temporal distance between spline points
      double time_shift;
      if (num_spline_points > extra_points) {
        time_shift = mju_max(time_horizon /
                              (num_spline_points - extra_points), 1.0e-5);
      } else {
        // not a valid setting, but avoid division by zero
        time_shift = time_horizon;
      }

      const std::shared_lock<std::shared_mutex> lock(mtx_);
      policy.plan.DiscardBefore(nominal_time);
      if (policy.plan.Size() == 0) {
        policy.plan.AddNode(time);
      }
      while (policy.plan.Size() < num_spline_points) {
        // duplicate the last node, with a time further in the future.
        double new_node_time = (policy.plan.end() - 1)->time() + time_shift;
        TimeSpline::Node new_node = policy.plan.AddNode(new_node_time);
        std::copy((policy.plan.end() - 2)->values().begin(),
                  (policy.plan.end() - 2)->values().end(),
                  new_node.values().begin());
      }
    } else {
      // non-sliding, resample the plan into a scratch plan
      double time_shift;
      if (interpolation_ == spline::SplineInterpolation::kZeroSpline) {
        time_shift = mju_max(time_horizon / num_spline_points, 1.0e-5);
      } else {
        time_shift = mju_max(time_horizon / (num_spline_points - 1), 1.0e-5);
      }

      // resample the nominal plan on a new set of spline points
      plan_scratch.Clear();
      plan_scratch.SetInterpolation(interpolation_);
      plan_scratch.Reserve(num_spline_points);

      // get spline points
      for (int t = 0; t < num_spline_points; t++) {
        TimeSpline::Node node = plan_scratch.AddNode(nominal_time);
        candidate_policy[winner].Action(node.values().data(), /*state=*/nullptr,
                                        nominal_time);
        nominal_time += time_shift;
      }

      // copy scratch into plan
      {
        const std::shared_lock<std::shared_mutex> lock(mtx_);
        policy.plan = plan_scratch;
      }
    }
  }

  // add random noise to nominal policy
  void AddNoiseToPolicy(double start_time, int i) {
    // start timer
    auto noise_start = std::chrono::steady_clock::now();

    // sampling token
    absl::BitGen gen_;

    // get standard deviation, fixed or mixture of noise_exploration[0,1]
    double std = noise_exploration[0];
    constexpr double kStd2Proportion = 0.2;  // hardcoded proportion of 2nd std
    if (noise_exploration[1] > 0 && absl::Bernoulli(gen_, kStd2Proportion)) {
      std = noise_exploration[1];
    }

    for (const TimeSpline::Node& node : candidate_policy[i].plan) {
      for (int k = 0; k < model->nu; k++) {
        double scale = 0.5 * (model->actuator_ctrlrange[2 * k + 1] -
                              model->actuator_ctrlrange[2 * k]);
        // Add noise
        node.values()[k] += absl::Gaussian<double>(gen_, 0.0, scale * std);;
      }
      Clamp(node.values().data(), model->actuator_ctrlrange, model->nu);
    }

    // end timer
    IncrementAtomic(noise_compute_time, GetDuration(noise_start));
  }

  // compute candidate trajectories
  void Rollouts(int num_trajectory, int horizon,
                ThreadPool& pool) {
    // reset noise compute time
    noise_compute_time = 0.0;

    // random search
    int count_before = pool.GetCount();
    for (int i = 0; i < num_trajectory; i++) {
      pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                    &state = this->state, &time = this->time,
                    &mocap = this->mocap, &userdata = this->userdata, horizon,
                    i]() {
        // copy nominal policy
        {
          const std::shared_lock<std::shared_mutex> lock(s.mtx_);
          s.candidate_policy[i].CopyFrom(s.policy, s.policy.num_spline_points);
        }

        // sample noise policy
        if (i != 0) s.AddNoiseToPolicy(time, i);

        // ----- rollout sample policy ----- //

        // policy
        auto sample_policy_i = [&candidate_policy = s.candidate_policy, &i](
                                  double* action, const double* state,
                                  double time) {
          candidate_policy[i].Action(action, state, time);
        };

        // policy rollout
        s.trajectory[i].Rollout(
            sample_policy_i, task, model, s.data_[ThreadPool::WorkerId()].get(),
            state.data(), time, mocap.data(), userdata.data(), horizon);
      });
    }
    pool.WaitCount(count_before + num_trajectory);
    pool.ResetCount();
  }

  // return trajectory with best total return
  const Trajectory* BestTrajectory() override {
    return winner >= 0 ? &trajectory[winner] : nullptr;
  }

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override {
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

    // sample traces
    for (int k = 0; k < num_trajectory_; k++) {
      // skip winner
      if (k == winner) continue;

      // plot sample
      for (int i = 0; i < best->horizon - 1; i++) {
        if (scn->ngeom + task->num_trace > scn->maxgeom) break;
        for (int j = 0; j < task->num_trace; j++) {
          // initialize geometry
          mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3, zero9,
                      color);

          // make geometry
          mjv_makeConnector(
              &scn->geoms[scn->ngeom], mjGEOM_LINE, width,
              trajectory[k].trace[3 * task->num_trace * i + 3 * j],
              trajectory[k].trace[3 * task->num_trace * i + 1 + 3 * j],
              trajectory[k].trace[3 * task->num_trace * i + 2 + 3 * j],
              trajectory[k].trace[3 * task->num_trace * (i + 1) + 3 * j],
              trajectory[k].trace[3 * task->num_trace * (i + 1) + 1 + 3 * j],
              trajectory[k].trace[3 * task->num_trace * (i + 1) + 2 + 3 * j]);

          // increment number of geometries
          scn->ngeom += 1;
        }
      }
    }
  }

  // planner-specific GUI elements
  void GUI(mjUI& ui) override {
    mjuiDef defSampling[] = {
        {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
        {mjITEM_SELECT, "Spline", 2, &interpolation_,
        "Zero\nLinear\nCubic"},
        {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
        {mjITEM_SLIDERNUM, "Noise Std", 2, noise_exploration, "0 1"},
        {mjITEM_SLIDERNUM, "Noise Std2", 2, noise_exploration+1, "0 1"},
        {mjITEM_CHECKBYTE, "Sliding plan", 2, &sliding_plan_, ""},
        {mjITEM_END}};

    // set number of trajectory slider limits
    mju::sprintf_arr(defSampling[0].other, "%i %i", 1, kMaxTrajectory);

    // set spline point limits
    mju::sprintf_arr(defSampling[2].other, "%i %i", MinSamplingSplinePoints,
                    MaxSamplingSplinePoints);

    // set noise standard deviation limits
    mju::sprintf_arr(defSampling[3].other, "%f %f", MinNoiseStdDev,
                    MaxNoiseStdDev);

    // add sampling planner
    mjui_add(&ui, defSampling);
  }

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
            int planner_shift, int timer_shift, int planning,
            int* shift) override {
    // ----- planner ----- //
    double planner_bounds[2] = {-6.0, 6.0};

    // improvement
    mjpc::PlotUpdateData(fig_planner, planner_bounds,
                        fig_planner->linedata[0 + planner_shift][0] + 1,
                        mju_log10(mju_max(improvement, 1.0e-6)), 100,
                        0 + planner_shift, 0, 1, -100);

    // legend
    mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Improvement");

    fig_planner->range[1][0] = planner_bounds[0];
    fig_planner->range[1][1] = planner_bounds[1];

    // bounds
    double timer_bounds[2] = {0.0, 1.0};

    // ----- timer ----- //

    PlotUpdateData(fig_timer, timer_bounds,
                  fig_timer->linedata[0 + timer_shift][0] + 1,
                  1.0e-3 * noise_compute_time * planning, 100,
                  0 + timer_shift, 0, 1, -100);

    PlotUpdateData(fig_timer, timer_bounds,
                  fig_timer->linedata[1 + timer_shift][0] + 1,
                  1.0e-3 * rollouts_compute_time * planning, 100,
                  1 + timer_shift, 0, 1, -100);

    PlotUpdateData(fig_timer, timer_bounds,
                  fig_timer->linedata[2 + timer_shift][0] + 1,
                  1.0e-3 * policy_update_compute_time * planning, 100,
                  2 + timer_shift, 0, 1, -100);

    // legend
    mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Noise");
    mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Rollout");
    mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Policy Update");

    // planner shift
    shift[0] += 1;

    // timer shift
    shift[1] += 3;
  }

  double CandidateScore(int candidate) const override {
    return trajectory[trajectory_order[candidate]].total_return;
  }

  // set action from candidate policy
  void ActionFromCandidatePolicy(double* action_, int candidate_,
                                const double* state_,
                                double time_) override {
    candidate_policy[trajectory_order[candidate_]].Action(action_, state_, time_);
  }

  void CopyCandidateToPolicy(int candidate) override {
    // set winner
    winner = trajectory_order[candidate];

    {
      const std::shared_lock<std::shared_mutex> lock(mtx_);
      previous_policy = policy;
      policy = candidate_policy[winner];
    }
  }

  // return number of parameters optimized by planner
  int NumParameters() override {
    return policy.num_spline_points * model->nu;
  };

  // ----- members ----- //
  mjModel* model;
  const Task* task;

  // state
  std::vector<double> state;
  double time;
  std::vector<double> mocap;
  std::vector<double> userdata;

  // policy
  SamplingPolicy policy;  // (Guarded by mtx_)
  SamplingPolicy candidate_policy[kMaxTrajectory];
  SamplingPolicy previous_policy;

  // scratch
  mjpc::spline::TimeSpline plan_scratch;

  // trajectories
  Trajectory trajectory[kMaxTrajectory];

  // order of indices of rolled out trajectories, ordered by total return
  std::vector<int> trajectory_order;

  // ----- noise ----- //
  double noise_exploration[2] = {0};  // stds for sampling: N(0, exploration)
  std::vector<double> noise;
mjpc::spline::SplineInterpolation interpolation_ =
      mjpc::spline::SplineInterpolation::kZeroSpline;

  // best trajectory
  int winner;

  // improvement
  double improvement;

  // flags
  int processed_noise_status;

  // timing
  std::atomic<double> noise_compute_time;
  double rollouts_compute_time;
  double policy_update_compute_time;

  // If true, use sliding plans (no resampling)
  std::uint8_t sliding_plan_ = false;

  int num_trajectory_;
  mutable std::shared_mutex mtx_;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_SAMPLING_PLANNER_H_
