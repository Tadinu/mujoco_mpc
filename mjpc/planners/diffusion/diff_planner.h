#ifndef MJPC_PLANNERS_DIFFUSION_PLANNER_H_
#define MJPC_PLANNERS_DIFFUSION_PLANNER_H_

#include <mujoco/mujoco.h>

#include <atomic>
#include <shared_mutex>
#include <vector>

#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/planners/diffusion/diff_config.h"
#include "mjpc/planners/diffusion/diff_core.h"

namespace mjpc {
class DiffusionPlanner : public Planner {
public:
  // constructor
  DiffusionPlanner() = default;

  // destructor
  ~DiffusionPlanner() override = default;

  // ----- methods ----- //

  // initialize data and settings
  void Initialize(mjModel* model, const Task& task) override;

  // allocate memory
  void Allocate() override;

  // reset memory to zeros
  void Reset(int horizon, const double* initial_repeated_action = nullptr) override;

  // set state
  void SetState(const State& state) override;

  // resize rollout data list
  void ResizeMjData(const mjModel* model, int num_threads) override;

  std::function<void()> post_resize_mjdata_cb_ = nullptr;

  void SetPostResizeMjData(const std::function<void()>& cb) { post_resize_mjdata_cb_ = cb; }

  // optimize nominal policy using random sampling
  SamplingPolicy PolicyFromY(int horizon, const Eigen::MatrixXd& Y0s);
  void Plan(int horizon, ThreadPool& pool);
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override;
  void NominalTrajectory(int horizon);

  // set action from policy
  void ActionFromPolicy(double* action, const double* state, double time, bool use_previous = false) override;

  // resample nominal policy
  void ResamplePolicy(int horizon);

  // add noise to nominal policy
  void AddNoiseToPolicy(int i, double std_min);

  // compute candidate trajectories
  void Rollouts(int num_trajectory, const std::vector<Eigen::MatrixXd>& Y0s, int horizon, ThreadPool& pool);

  // return trajectory with best total return
  const Trajectory* BestTrajectory() override;

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override;

  // planner-specific GUI elements
  void GUI(mjUI& ui) override;

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift, int timer_shift, int planning,
             int* shift) override;

  // return number of parameters optimized by planner
  int NumParameters() override { return policy.num_spline_points * action_dim_; };

  // ----- members ----- //
  mjModel* model;
  const Task* task;

  // state
  std::vector<double> state;
  double time;
  std::vector<double> mocap;
  std::vector<double> userdata;

  // policy
  SamplingPolicy policy; // (Guarded by mtx_)
  SamplingPolicy candidate_policy[kMaxTrajectory];
  SamplingPolicy nominal_policy;
  SamplingPolicy previous_policy;

  // scratch
  std::vector<double> parameters_scratch; // [action_dim_]
  std::vector<double> times_scratch; // [action_dim_]
  void UpdatePolicyWithScratch(SamplingPolicy& in_policy);

  // trajectories
  TrajectoryPtr nominal_trajectory = std::make_shared<Trajectory>();

  // order of indices of rolled out trajectories, ordered by total return
  std::vector<int> trajectory_order;

  // ----- noise ----- //
  double std_initial_; // standard deviation for sampling normal: N(0,
  // std)
  double std_min_; // the minimum allowable std
  double explore_fraction_ = 0; // fraction of trajectories that will use
  // std_initial instead of the variance from CEM
  std::vector<double> noise;
  std::vector<double> variance;

  // number of elite samples
  int n_elite_;

  // improvement
  double improvement;

  // timing
  std::atomic<double> noise_compute_time;
  double rollouts_compute_time;
  double policy_update_compute_time;

  mjpc::spline::SplineInterpolation interpolation_ = mjpc::spline::SplineInterpolation::kZeroSpline;
  int num_trajectory_;
  mutable std::shared_mutex mtx_;

  // DIFFUSION
  DiffusionConfig diff_config_;
  std::unique_ptr<MBDPI> mbdpi_ = nullptr;
  Eigen::MatrixXd Y_; // (nodes_num, nu)
  double ctrl_dt_ = 0.01;
  std::chrono::steady_clock::time_point last_plan_time_;

  Eigen::MatrixXd Shift(const Eigen::MatrixXd& Y0, // (nodes_num, nu)
                        double shift_time,
                        const Eigen::VectorXd& step_nodes) const {
    int nodes_num = Y0.rows();
    int nu = Y0.cols();

    Eigen::VectorXd shifted_nodes = step_nodes.array() + shift_time;
    Eigen::MatrixXd Y_new(nodes_num, nu);
    for (int j = 0; j < nu; ++j) {
      // Eigen splines have shape (1, num_points)
      Eigen::RowVectorXd values = Y0.col(j).transpose();
      Eigen::Spline<double, 1> spline = Eigen::SplineFitting<Eigen::Spline<double, 1>>::Interpolate(
          values, 2, step_nodes // degree-2 spline
          );
      for (int i = 0; i < nodes_num; ++i)
        Y_new(i, j) = spline(shifted_nodes(i))(0);
    }
    return Y_new;
  }

  Eigen::MatrixXd Shift(const Eigen::MatrixXd& Y0, double shift_time) const {
    return Shift(Y0, shift_time, mbdpi_->step_nodes);
  }

  void ReverseScan(Eigen::MatrixXd& Y0, const Eigen::VectorXd& traj_diffuse_factors,
                   int horizon,
                   ThreadPool& pool);
};
} // namespace mjpc

#endif  // MJPC_PLANNERS_DIFFUSION_PLANNER_H_
