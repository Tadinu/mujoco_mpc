#include "mjpc/optimizers/second_order_optimizer.h"

// LBFGSB
#include <LBFGSB.h>

#include "mjpc/planners/cio/cio_planner.h"

double SecondOrderOptimizer::operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad) {
#if 0
  const auto& model_grad = mj_planner_->model_derivative.C;
  grad.resize(model_grad.size());
  memcpy(grad.data(), model_grad.data(), model_grad.size() * sizeof(double));
  return mj_planner_->BestTrajectory()->total_return;
#else
  auto* cio_planner = dynamic_cast<mjpc::CIOPlanner*>(mj_planner_);
  if (!cio_planner) {
    return 0;
  }

  // CIOObservation obs;
  // obs.from_data(x.data(), (x.size() - CIOObservation::pose_vel_acc_size) / CIOObservation::contact_size);
  const double last_cost = cio_planner->RolloutNominalTrajectory(x);
#if 1
  const auto cost_derivative = cio_planner->ComputeDerivatives(cio_planner->nominal_trajectory);
  const auto& cost_grad = cost_derivative.cu;
  grad.noalias() = Eigen::VectorXd::Map(cost_grad.data(), cost_grad.size());
#else
  const auto f = [this, last_cost](const autodiff::ArrayXreal& d) -> autodiff::real { return last_cost; };
  autograd(f, x, &grad);
#endif
  return last_cost;
#endif
}

Eigen::VectorXd SecondOrderOptimizer::optimize(const mjpc::TrajectoryPtr& trajectory, int policy_idx,
                                               int thread_worker_id) {
  auto* cio_planner = dynamic_cast<mjpc::CIOPlanner*>(mj_planner_);
  if (!cio_planner) {
    return x_;
  }
  LBFGSpp::LBFGSBParam<double> param;
  LBFGSpp::LBFGSBSolver<double> solver(param);

  // Optimize
  {
    // Update [traj_, goals, x_, stage_idx_]
    // Calculate [x_]
    // if (START_STAGE == stage_idx_)
    {
      const std::vector<double> init = cio_planner->GetNominalPolicyValues(true);
      x_ = Eigen::VectorXd::Map(init.data(), init.size());
    }

    // Variable bounds
    const int n = x_.size();
    if (n == 0) {
      return x_;
    }
    Eigen::VectorXd lb = Eigen::VectorXd::Constant(n, -1);  // lower
    Eigen::VectorXd ub = Eigen::VectorXd::Constant(n, 1);   // upper

    // Invoke operator(), optimizing batch of [stage_]
    double fx;
    int niter = solver.minimize(*this, x_, fx, lb, ub);

    std::cout << niter << " iterations" << std::endl;
    std::cout << "x_ = \n" << x_.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;
    std::cout << "grad = " << solver.final_grad().transpose() << std::endl;
    std::cout << "projected grad norm = " << solver.final_grad_norm() << std::endl;
    return x_;
  }
}