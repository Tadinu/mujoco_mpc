#pragma once

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include <Eigen/Core>
#include <cassert>
#include <chrono>
#include <map>
#include <memory>
#include <shared_mutex>
#include <vector>

// mjpc
#include <Eigen/src/Core/DenseBase.h>

#include "mjpc/optimizers/riemannian_optimizer.h"
#include "mjpc/planners/cio/cio_common.h"
#include "mjpc/planners/cio/cio_util.h"
#include "mjpc/planners/gradient/planner.h"
#include "mjpc/planners/planner.h"
#include "mjpc/tasks/mpl/mpl_cost.h"
#include "mjpc/utilities.h"

namespace mjpc {
using mjpc::spline::SplineInterpolation;
using mjpc::spline::TimeSpline;

class CIOPlanner : public mjpc::GradientPlanner {
public:
  CIOPlanner() {}
  ~CIOPlanner() override = default;
  // =========================================================================================================
  // MJPC-PLANNER IMPL --
  //
  void Initialize(mjModel* model, const mjpc::Task& _task) override {
    GradientPlanner::Initialize(model, _task);

    // Init task CIO
    if (task->IsCIOSupported()) {
      InitTaskCIO();
    }
  }

  void InitTaskCIO() {
    optimizer_ = std::make_shared<RiemannianOptimizer>(model, task->data_, const_cast<Task*>(task), this);
  }

  BaseOptimizerPtr optimizer() const { return optimizer_; }
  void Plan(int horizon, mjpc::ThreadPool& pool) {
#if CIO_USE_OPTIMIZER
    horizon_ = horizon;
    pool_ = &pool;

    if (optimizer_) {
      // Optimize [nominal_policy] by rolling out [nominal_trajectory] here-in
      optimizer_->optimize();

      // [opt_vals()] -> [nominal_policy]'s parameters
      {
        const std::shared_lock<std::shared_mutex> lock(mtx_);
#if CIO_USE_ACTION_SPLINE
        // parameters
        mju_copy(nominal_policy.parameters.data(), optimizer_->opt_vals().data(), policy.num_parameters);
#else
        mju_copy(action_.data(), optimizer_->opt_vals().data(), policy.num_parameters);

#endif
      }
    }
#endif
  }

  std::vector<double> GetNominalPolicyValues(bool with_noise) override {
#if 1
    // Resample [nominal_policy]
    ResamplePolicy(horizon_);
    if (with_noise) {
      // Perturb [nominal_policy]
      AddNoiseToPolicy(nominal_policy);
    }
    return nominal_policy.parameters;
#else
    return nominal_trajectory->actions;
#endif
  }

  double RolloutNominalTrajectory(const Eigen::VectorXd& x) {
    // 0- [parameters_scratch], [times_scratch]
    ResamplePolicy(horizon_);
    parameters_scratch = std::vector<double>(x.data(), x.data() + x.size());

    // 1- Update [nominal_policy] with [times_scratch] + [parameters_scratch] of all [candidate_policy[]]
    const std::unique_lock<std::shared_mutex> lock(mtx_);
    {
      mju_copy(nominal_policy.parameters.data(), parameters_scratch.data(), nominal_policy.num_parameters);
      mju_copy(nominal_policy.times.data(), times_scratch.data(), nominal_policy.num_spline_points);
    }

    // 2- Rollout [nominal_policy] by [nominal_trajectory]
    auto frun_nominal_policy = [&cp = nominal_policy](double* action, const double* state, double time) {
      cp.Action(action, state, time);
    };
    nominal_trajectory->Rollout(frun_nominal_policy, task, model, data_[0].get(), state.data(), time,
                                mocap.data(), userdata.data(), horizon_);

    return nominal_trajectory->total_return;
  }

  // =========================================================================================================
  // MJPC-PLANNER IMPL --
  //
  void Allocate() override { GradientPlanner::Allocate(); }

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override {
    GradientPlanner::Traces(scn);
#if 0
    static constexpr float RED[] = {1.0, 0.0, 0.0, 1.0};
    auto scene = scn ? scn : task->scene_;
    for (const auto& i : trajectory) {
      const auto cio_traj = std::dynamic_pointer_cast<MPLCostCalculator>(i);
      if (!cio_traj) {
        continue;
      }

      for (const auto& [_, contact_state_list] : cio_traj->GetContactStates()) {
        for (const auto& contact : contact_state_list) {
          AddConnector(scene, mjGEOM_LINE, 2.5, contact.r.data(), contact.pi_H_.data(), RED);
        }
      }

      for (const auto& [_, cio_obj] : cio_traj->GetAllObjects()) {
        const auto cuboid_obj = std::dynamic_pointer_cast<CIOCuboid>(cio_obj);
        if (!cuboid_obj) {
          continue;
        }

        // AABB
        const int cuboid_id = cuboid_obj->id();
        const auto* pos = task->QueryBodyPos(cuboid_id);
        double mat[9];
        mju_quat2Mat(mat, task->QueryBodyQuat(cuboid_id));
        float rgba[4] = {0, 1, 0, 0.5};
        AddGeom(scene, mjGEOM_BOX, task->QueryGeomSize("object").data(), pos, mat, rgba);

        // Vertices
        for (const auto& obj_vert : cuboid_obj->vertices()) {
          static constexpr float BLUE[] = {0.0, 0.0, 1.0, 1.0};
          AddGeom(scene, mjGEOM_SPHERE, (mjtNum[]){0.003}, obj_vert.data(), /*mat=*/nullptr, BLUE);
        }
      }
    }
#endif
  }

  void ClearTrace() override { GradientPlanner::ClearTrace(); }

  // planner-specific GUI elements
  void GUI(mjUI& ui) override { GradientPlanner::GUI(ui); }

  // optimize nominal policy
  void OptimizePolicy(int horizon, mjpc::ThreadPool& pool) override {
#if CIO_USE_OPTIMIZER
    ResizeMjData(model, pool.NumThreads());

    // maximum number of trajectories in linesearch
    num_trajectory = mju_min(num_trajectory, kMaxTrajectory);

    // copy [policy] (as the latest best) -> [nominal_policy]
    policy.num_parameters = model->nu * policy.num_spline_points;
    {
      const std::shared_lock<std::shared_mutex> lock(mtx_);
      nominal_policy.CopyFrom(policy, policy.num_spline_points);
    }

    // previous best cost
    double c_prev = nominal_trajectory->total_return;
    double c_best = c_prev;

    // CIO plan optimizing, rolling out [nominal_trajectory] here-in
    Plan(horizon, pool);

#if CIO_USE_BATCH_GRADIENT
    // Rollout [candidate_policy[]]
    for (int r = 0; r < settings.max_rollout; r++) {
      // copy [nominal_policy] -> [candidate_policy[k]]
      for (int k = 1; k < num_trajectory; k++) {
        candidate_policy[k].CopyFrom(nominal_policy, nominal_policy.num_spline_points);
      }

      // rollout all of [trajectory[]] on corresponding [candidate_policy[]] (parallel)
      int count_before = pool.GetCount();
      for (int i = 0; i < num_trajectory; i++) {
        pool.Schedule([&data = data_, &trajectory_i = trajectory[i],
                       &candidate_policy_i = candidate_policy[i], &model = this->model, &task = this->task,
                       &state = this->state, &time = this->time, &mocap = this->mocap, horizon,
                       &userdata = this->userdata]() {
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

      // ----- evaluate rollouts ------ //
      winner = num_trajectory - 1;
      for (int j = num_trajectory - 1; j >= 0; j--) {
        // compute cost
        double c_sample = trajectory[j]->total_return;

        // compare cost
        if (c_sample < c_best) {
          c_best = c_sample;
          winner = j;
        }
      }

      // update nominal with winner
      if (winner != 0) {
        nominal_policy.CopyParametersFrom(winner_policy().parameters, winner_policy().times);
        nominal_trajectory = trajectory[winner];
      }

      // improvement
      action_step = linesearch_steps[winner];
      expected = -action_step * (gradient.dV[0]) - 1.0e-16;
      improvement = c_prev - c_best;
      surprise = mju_min(mju_max(0, improvement / expected), 2);
    }

    // check for improvement
    if (c_best >= c_prev) {
      winner = num_trajectory - 1;
    }
#endif
    // copy [winner_policy()] -> [policy] to be used in [ActionFromPolicy()]
    {
      const std::shared_lock<std::shared_mutex> lock(mtx_);
      previous_policy = policy;
#if CIO_USE_BATCH_GRADIENT
      policy.CopyParametersFrom(winner_policy().parameters, winner_policy().times);
#else
      policy.CopyParametersFrom(nominal_policy.parameters, nominal_policy.times);
#endif
    }
#else
    // Rollout policies & calculate [trajectories' total_return], including [nominal_trajectory]
    GradientPlanner::OptimizePolicy(horizon, pool);
#endif
  }

  void ComputeDerivatives() {
    // compute model and sensor Jacobians from [nominal_trajectory] rolled out above
    model_derivative.Compute(model, data_, nominal_trajectory->states.data(),
                             nominal_trajectory->actions.data(), nominal_trajectory->times.data(), dim_state,
                             dim_state_derivative, dim_action, dim_sensor, horizon_, settings.fd_tolerance,
                             settings.fd_mode, *pool_, false);

    // compute cost derivatives from [nominal_trajectory] & [model_derivative]
    cost_derivative.Compute(nominal_trajectory->residual.data(), model_derivative.C.data(),
                            model_derivative.D.data(), dim_state_derivative, dim_action, dim_max, dim_sensor,
                            task->num_residual, task->dim_norm_residual.data(), task->num_term,
                            task->weight.data(), task->norm.data(), task->norm_parameter.data(),
                            task->num_norm_parameter.data(), task->risk, horizon_, *pool_);
  }

  void AddNoiseToPolicy(GradientPolicy& in_policy) {
    // sampling token
    absl::BitGen gen_;

    // get standard deviation, fixed or mixture of noise_exploration[0,1]
    double std = noise_exploration[0];
    constexpr double kStd2Proportion = 0.2;  // hardcoded proportion of 2nd std
    if (noise_exploration[1] > 0 && absl::Bernoulli(gen_, kStd2Proportion)) {
      std = noise_exploration[1];
    }

    for (auto t = 0; t < in_policy.num_spline_points; ++t) {
      for (int k = 0; k < model->nu; ++k) {
        double scale = 0.5 * (model->actuator_ctrlrange[2 * k + 1] - model->actuator_ctrlrange[2 * k]);
        double noise = absl::Gaussian<double>(gen_, 0.0, scale * std);
        in_policy.parameters[t * model->nu + k] += noise;
      }
      Clamp(in_policy.parameters.data(), model->actuator_ctrlrange, model->nu);
    }
  }

  const Trajectory* BestTrajectory() override { return nominal_trajectory.get(); }

  // set action from policy
  void ActionFromPolicy(double* action, const double* state, double time, bool use_previous) override {
#if CIO_USE_ACTION_SPLINE
    GradientPlanner::ActionFromPolicy(action, state, time, use_previous);
#else
    const std::shared_lock<std::shared_mutex> lock(policy_mutex_);

    // WAIT ACTION TO BE COMPUTED
    if (action_.empty()) {
      return;
    }

    // APPLY ACTION: COPY [action_] -> [action]
    MJPC_PRINTDB("ACTION", action_);
    mju_copy(action, action_.data(), int(action_.size()));
    // Clear [action_]
    action_.clear();

    // Clamp controls on outputted [action]
    mjpc::Clamp(action, model->actuator_ctrlrange, model->nu);
#endif
  }

private:
  BaseOptimizerPtr optimizer_ = nullptr;

  int horizon_ = 0;
  mjpc::ThreadPool* pool_ = nullptr;

  // noise
  double noise_exploration[2] = {0, 0.01};  // stds for sampling: N(0, exploration)
  std::vector<double> noise;
  mjpc::spline::SplineInterpolation interpolation_ = mjpc::spline::SplineInterpolation::kZeroSpline;

  // mjpc
  mutable std::shared_mutex policy_mutex_;
  // [action_] is shared among policy motion planning threads.
  std::vector<double> action_ = std::vector<double>(6);
};
}  // end namespace mjpc
