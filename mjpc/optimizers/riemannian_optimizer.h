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
#include "mjpc/planners/planner.h"
#include "mjpc/task.h"

using namespace std;
using namespace Optimization;
using namespace Riemannian;

#define DATA_SIZE (20)

class RiemannianOptimizer : public BaseOptimizer {
public:
  RiemannianOptimizer(mjModel *mj_model, mjData *mj_data, mjpc::Task *mj_task, mjpc::Planner *mj_planner)
      : BaseOptimizer(mj_model, mj_data, mj_task, mj_planner) {}

  void optimize() override;
  Eigen::VectorXd project_to_tangent_space(const Eigen::VectorXd &X, const Eigen::VectorXd &V);
  std::vector<double> opt_vals() const override { return std::vector(x_.data(), x_.data() + x_.size()); }

private:
  Eigen::VectorXd x_ = Eigen::VectorXd::Zero(DATA_SIZE);
};
using RiemannianOptimizerPtr = std::shared_ptr<RiemannianOptimizer>;