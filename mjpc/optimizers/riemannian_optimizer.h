#pragma once

#include <time.h>

#include <fstream>

#include "mjpc/optimizers/base_optimizer.h"
// MuJoCo
#include <mujoco/mujoco.h>

// Eigen
#include <Eigen/Core>

// MJPC

#include "mjpc/optimizers/riemannian_opt/Riemannian/GradientDescent.h"
#include "mjpc/optimizers/riemannian_opt/Riemannian/TNT.h"
#include "mjpc/planners/gradient/policy.h"
#include "mjpc/planners/planner.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"

using namespace std;
using namespace Optimization;
using namespace Riemannian;

class RiemannianOptimizer : public BaseOptimizer {
public:
  RiemannianOptimizer(mjModel *mj_model, mjData *mj_data, mjpc::Task *mj_task, mjpc::Planner *mj_planner,
                      int data_dim)
      : BaseOptimizer(mj_model, mj_data, mj_task, mj_planner), data_dim_(data_dim) {}

  Eigen::VectorXd optimize(const mjpc::TrajectoryPtr &trajectory, int policy_idx,
                           int thread_worker_id) override;
  Eigen::VectorXd project_to_tangent_space(const Eigen::VectorXd &X, const Eigen::VectorXd &V);

private:
  int data_dim_ = 0;
};
using RiemannianOptimizerPtr = std::shared_ptr<RiemannianOptimizer>;