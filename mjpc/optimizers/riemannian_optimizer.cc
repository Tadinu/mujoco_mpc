#include "mjpc/optimizers/riemannian_optimizer.h"

#include "mjpc/planners/cio/cio_planner.h"

typedef double Scalar;

/// Utility function: projection of the vector V \in R^3 onto the tangent
/// space T_X(S^2) of S^2 at the point
Eigen::VectorXd RiemannianOptimizer::project_to_tangent_space(const Eigen::VectorXd& X,
                                                              const Eigen::VectorXd& V) {
  return V - X.dot(V) * X;
}

Eigen::VectorXd RiemannianOptimizer::optimize(const mjpc::TrajectoryPtr& trajectory, int policy_idx,
                                              int thread_worker_id) {
  /// SET UP OPTIMIZATION

  /// In the example that follows, we will treat the point P as a fixed (i.e.
  /// unoptimized) parameter of the objective f, and pass this to the TNT
  /// optimization library as a user-supplied optional argument.

  /// SET UP FUNCTION HANDLES
  const auto P = Eigen::VectorXd::Zero(data_dim_);

  /// Objective
  Objective<Eigen::VectorXd, Scalar> f_objective = [this, trajectory, policy_idx,
        thread_worker_id](const Eigen::VectorXd& X) -> double {
    auto* cio_planner = dynamic_cast<mjpc::CIOPlanner*>(mj_planner_);
    if (!cio_planner) {
      return 0;
    }

    // Euclidean gradient
    mjpc::GradientPolicy& policy = cio_planner->candidate_policy[policy_idx];
    mju_copy(policy.parameters.data(), X.data(), X.size());
    return cio_planner->RolloutTrajectory(trajectory, policy_idx, thread_worker_id);
  };

  /// Gradient
  VectorField<VectorXd, VectorXd> grad_f = [this, trajectory](const VectorXd& X) -> VectorXd {
    auto* cio_planner = dynamic_cast<mjpc::CIOPlanner*>(mj_planner_);
    if (!cio_planner) {
      return VectorXd::Zero(data_dim_);
    }

    // Euclidean gradient
    const auto cost_derivative = cio_planner->ComputeDerivatives(trajectory);

    const auto& cost_grad = cost_derivative.cu;
    const auto nabla_f = VectorXd::Map(cost_grad.data(), cost_grad.size());

    // Compute Riemannian gradient from Euclidean one
    return project_to_tangent_space(X, nabla_f);
  };

  /// Riemannian Hessian constructor: Returns the Riemannian Hessian operator
  /// H(X): T_X(S^2) -> T_X(S^2) at X
  LinearOperatorConstructor<VectorXd, VectorXd> HC = [this, &grad_f](const VectorXd& X) {
    // Euclidean Hessian matrix
    auto EucHess = 2 * MatrixXd::Identity(data_dim_, data_dim_);

    // Return Riemannian Hessian-vector product operator using the
    // Euclidean Hessian
    LinearOperator<VectorXd, VectorXd> Hessian = [this, EucHess, &grad_f](const VectorXd& X,
      const VectorXd& Xdot) -> VectorXd {
      return project_to_tangent_space(X, EucHess * Xdot) - X.dot(grad_f(X)) * Xdot;
    };

    return Hessian;
  };

  /// Riemannian metric on S^2: this is just the usual inner-product on R^3
  RiemannianMetric<VectorXd, VectorXd, Scalar> metric = [
      ](const VectorXd& X, const VectorXd& V1, const VectorXd& V2) {
    return V1.dot(V2);
  };

  /// Projection-based retraction operator for S^2
  Retraction<VectorXd, VectorXd> retract = [](const VectorXd& X, const VectorXd& V) {
    return (X + V).normalized();
  };

  /// SET INITIAL POINT
  auto* cio_planner = dynamic_cast<mjpc::CIOPlanner*>(mj_planner_);
  if (!cio_planner) {
    return Eigen::VectorXd::Zero(data_dim_);
  }
  const std::vector<double> init = cio_planner->GetCandidatePolicyValues(policy_idx, true);
  const auto x0 = Eigen::VectorXd::Map(init.data(), init.size());

  // cout << "X0 = " << endl << x0 << endl << endl;

#if 1
  /// RUN GRADIENT DESCENT OPTIMIZER!
  // Set gradient descent options
  GradientDescentParams<Scalar> gd_params;
#if 0
  gd_params.max_iterations = 1;
  gd_params.max_ls_iterations = 1;
#endif
  gd_params.verbose = false;

  GradientDescentResult<VectorXd, Scalar> gd_result =
      GradientDescent<VectorXd, VectorXd, Scalar>(f_objective, grad_f, metric, retract, x0, gd_params);

  //cout << "FINISHED GRADIENT DESCENT OPTIMIZER! - THREAD ID " << thread_worker_id << endl << endl;

  //cout << "Gradient descent estimate x_final:  " << endl << gd_result.x << endl << endl;
  return gd_result.x;
#else
  /// RUN TNT OPTIMIZER!
  //cout << "RUNNING TNT OPTIMIZER! - THREAD ID " << thread_worker_id << endl << endl;

  // Set TNT options
  TNTParams<Scalar> tnt_params;
  tnt_params.verbose = true;

  auto tnt_result = TNT<VectorXd, VectorXd, Scalar>(f_objective, grad_f, HC, metric, retract, x0, tnt_params);
  cout << "Truncated-Newton trust-region estimate x_final:  " << endl << tnt_result.x << endl << endl;
  return tnt_result.x;
#endif
}
