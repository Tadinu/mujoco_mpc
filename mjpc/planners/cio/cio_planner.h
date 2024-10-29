#pragma once

#include <mujoco/mujoco.h>

#include <Eigen/Core>
#include <cassert>
#include <chrono>
#include <map>
#include <memory>
#include <shared_mutex>
#include <vector>

// LBFGSB
#include <LBFGSB.h>

// mjpc
#include "mjpc/planners/cio/cio_common.h"
#include "mjpc/planners/cio/cio_trajectory.h"
#include "mjpc/planners/cio/cio_util.h"
#include "mjpc/planners/cross_entropy/planner.h"
#include "mjpc/utilities.h"

class CIOptimizer {
public:
  static constexpr int START_STAGE = 0;
  CIOptimizer() = default;
  CIOptimizer(mjModel* mj_model, mjData* mj_data, mjpc::Task* mj_task)
      : mj_model_(mj_model), mj_data_(mj_data), mj_task_(mj_task) {}

  Eigen::VectorXd opt_x() const { return x_; }
  double cost() const { return 0; }
  double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
    CIOObservation obs;
    obs.from_data(x.data(), (x.size() - CIOObservation::pose_vel_acc_size) / CIOObservation::contact_size);
    // Calculate cost()
    return cost();
  }

  void optimize() {
    LBFGSpp::LBFGSBParam<double> param;
    LBFGSpp::LBFGSBSolver<double> solver(param);

    // Optimize
    {
      // Update [traj_, goals, x_, stage_idx_]
#if 0
      // Calculate [x_]
      if (START_STAGE == stage_idx_) {
        const std::vector<double> init = traj->GetObservationsData();
        x_ = Eigen::VectorXd::Map(init.data(), init.size());
      }
#endif

      // Variable bounds
      const int n = x_.size();
      Eigen::VectorXd lb = Eigen::VectorXd::Constant(n, 0);  // lower
      Eigen::VectorXd ub = Eigen::VectorXd::Constant(n, 1);  // upper

      // Invoke operator(), optimizing batch of [stage_]
      double fx;
      int niter = solver.minimize(*this, x_, fx, lb, ub);

      std::cout << niter << " iterations" << std::endl;
      std::cout << "x_ = \n" << x_.transpose() << std::endl;
      std::cout << "f(x) = " << fx << std::endl;
      std::cout << "grad = " << solver.final_grad().transpose() << std::endl;
      std::cout << "projected grad norm = " << solver.final_grad_norm() << std::endl;
    }
  }

private:
  mjModel* mj_model_ = nullptr;
  mjData* mj_data_ = nullptr;
  mjpc::Task* mj_task_ = nullptr;

  // cio
  int stage_idx_ = 0;
  Eigen::VectorXd x_;
};
using CIOptimizerPtr = std::shared_ptr<CIOptimizer>;

namespace mjpc {
using mjpc::spline::SplineInterpolation;
using mjpc::spline::TimeSpline;

class CIOPlanner : public mjpc::CrossEntropyPlanner {
public:
  CIOPlanner() = default;
  ~CIOPlanner() override = default;
  // =========================================================================================================
  // MJPC-PLANNER IMPL --
  //
  void Initialize(mjModel* model, const mjpc::Task& _task) override {
    CrossEntropyPlanner::Initialize(model, _task);

    // Init task CIO
    if (task->IsCIOSupported()) {
      InitTaskCIO();
    }
  }

  void InitTaskCIO() {
    optimizer_ = std::make_shared<CIOptimizer>(model, task->data_, const_cast<Task*>(task));
  }

  CIOptimizerPtr optimizer() const { return optimizer_; }
  void Plan() {
#if CIO_USE_LBFGSB
    // Loop optimizing over multiple stages
    optimizer_->optimize();
    // Copy [optimizer_->opt_x()] -> [action_]
    const auto opt = optimizer_->opt_x();
    // Finger0 vel
    action_[0] = opt[26];
    action_[1] = opt[27];
    action_[2] = opt[28];
    // Finger1 vel
    action_[3] = opt[45];
    action_[4] = opt[46];
    action_[5] = opt[47];
#endif
  }

  // =========================================================================================================
  // MJPC-PLANNER IMPL --
  //
  // init trajectories
  void InitTrajectory() override {
    for (auto& traj : trajectory) {
      traj = std::make_shared<CIOTrajectory>();
    }
  }

  void Allocate() override { CrossEntropyPlanner::Allocate(); }

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override {
    // CrossEntropyPlanner::Traces(scn);
#if 0
    static constexpr float RED[] = {1.0, 0.0, 0.0, 1.0};
    auto scene = scn ? scn : task->scene_;
    for (const auto& i : trajectory) {
      const auto cio_traj = std::dynamic_pointer_cast<CIOTrajectory>(i);
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

  void ClearTrace() override { CrossEntropyPlanner::ClearTrace(); }

  // planner-specific GUI elements
  void GUI(mjUI& ui) override { CrossEntropyPlanner::GUI(ui); }

  // optimize nominal policy
  void OptimizePolicy(int horizon, mjpc::ThreadPool& pool) override {
    CrossEntropyPlanner::OptimizePolicy(horizon, pool);
  }

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, mjpc::ThreadPool& pool) override {
    CrossEntropyPlanner::NominalTrajectory(horizon, pool);
  }

  // set action from policy
  void ActionFromPolicy(double* action, const double* state, double time, bool use_previous) override {
#if CIO_USE_LBFGSB
    const std::shared_lock<std::shared_mutex> lock(policy_mutex_);

    // WAIT ACTION TO BE COMPUTED
    if (action_.empty()) {
      return;
    }

    // APPLY ACTION: COPY [action_] -> [action]
    FAB_PRINTDB("ACTION", action_);
    mju_copy(action, action_.data(), int(action_.size()));
    // Clear [action_]
    action_.clear();

    // Clamp controls on outputted [action]
    mjpc::Clamp(action, model->actuator_ctrlrange, model->nu);
#else
    CrossEntropyPlanner::ActionFromPolicy(action, state, time, use_previous);
#endif
  }

private:
  CIOptimizerPtr optimizer_ = nullptr;

  // mjpc
  mutable std::shared_mutex policy_mutex_;
  // [action_] is shared among policy motion planning threads.
  std::vector<double> action_ = std::vector<double>(6);
};
}  // end namespace mjpc
