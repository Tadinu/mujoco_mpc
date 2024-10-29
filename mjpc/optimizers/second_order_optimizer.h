#pragma once

#include "mjpc/optimizers/base_optimizer.h"

// MuJoCo
#include <mujoco/mujoco.h>

// Eigen
#include <Eigen/Core>

// MJPC
#include "mjpc/planners/planner.h"
#include "mjpc/task.h"

// AUTODIFF
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

// MCMC
#define MCMC_ENABLE_EIGEN_WRAPPERS
#include "mcmc.hpp"

class SecondOrderOptimizer : public BaseOptimizer {
public:
  SecondOrderOptimizer(mjModel* mj_model, mjData* mj_data, mjpc::Task* mj_task, mjpc::Planner* mj_planner)
      : BaseOptimizer(mj_model, mj_data, mj_task, mj_planner) {}
  static constexpr int START_STAGE = 0;

  double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad);

  void optimize() override;
  std::vector<double> opt_vals() const override { return std::vector(x_.data(), x_.data() + x_.size()); }

  double autograd(const std::function<autodiff::real(const autodiff::ArrayXreal& d)>& f,
                  const Eigen::VectorXd& vals, Eigen::VectorXd* grad_out) {
    autodiff::real u;
    autodiff::ArrayXreal x = vals.eval();

    if (grad_out) {
      // NOTE: Must store the result as a temp here
      Eigen::VectorXd grad_tmp = autodiff::gradient(f, autodiff::wrt(x), autodiff::at(x), u);
      *grad_out = grad_tmp;
    } else {
      u = f(x);
    }

    return u.val();
  }

#if 0
  inline Eigen::VectorXd eigen_randn_colvec(size_t nr) {
    static std::mt19937 gen{std::random_device{}()};
    static std::normal_distribution<> dist;

    return Eigen::VectorXd{nr}.unaryExpr([&](double x) {
      (void)(x);
      return dist(gen);
    });
  }

  void autodiff(const std::function<autodiff::real(const autodiff::ArrayXreal& d)>& f) {
    const double mu = 2.0;
    const double sigma = 2.0;

    Eigen::VectorXd initial_vals(2);
    initial_vals(0) = mu + 1;     // mu
    initial_vals(1) = sigma + 1;  // sigma

    // Normal density log form
    autograd(f, initial_vals, &x_);
    std::cout << x_ << std::endl;
  }
#endif

private:
  // cio
  int stage_idx_ = 0;
  Eigen::VectorXd x_;
};
using SecondOrderOptimizerPtr = std::shared_ptr<SecondOrderOptimizer>;