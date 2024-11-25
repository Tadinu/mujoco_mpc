#include "mjpc/optimizers/riemannian_optimizer.h"

#include "mjpc/planners/cio/cio_planner.h"

typedef double Scalar;
typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;

/// Utility function: projection of the vector V \in R^3 onto the tangent
/// space T_X(S^2) of S^2 at the point
Eigen::VectorXd RiemannianOptimizer::project_to_tangent_space(const Eigen::VectorXd &X,
                                                              const Eigen::VectorXd &V) {
  return V - X.dot(V) * X;
}

void RiemannianOptimizer::optimize() {
  /// SET UP OPTIMIZATION

  ///  We will minimize the function f(X; P) := || X - P ||^2, where P in S^2 is
  ///  a fixed point on the sphere
  ///
  /// In the example that follows, we will treat the point P as a fixed (i.e.
  /// unoptimized) parameter of the objective f, and pass this to the TNT
  /// optimization library as a user-supplied optional argument.

  Vector P = Eigen::VectorXd::Unit(DATA_SIZE, DATA_SIZE - 1);  // = {0.0, 0.0, 1.0}; // P is the north pole

  /// SET UP FUNCTION HANDLES
  /// Objective
  Objective<Vector, Scalar, Vector> F = [this](const Vector &X, const Vector &P) -> double {
#if 1
    auto *cio_planner = dynamic_cast<mjpc::CIOPlanner *>(mj_planner_);
    if (!cio_planner) {
      return 0;
    }

    // Euclidean gradient
    return cio_planner->RolloutNominalTrajectory(X);
#else
    return (X - P).squaredNorm();
#endif
  };

  /// Gradient
  VectorField<Vector, Vector, Vector> grad_F = [this](const Vector &X, const Vector &P) -> Vector {
#if 1
    auto *cio_planner = dynamic_cast<mjpc::CIOPlanner *>(mj_planner_);
    if (!cio_planner) {
      return Vector::Zero(DATA_SIZE);
    }

    // Euclidean gradient
    cio_planner->RolloutNominalTrajectory(X);

    cio_planner->ComputeDerivatives();

    const auto &cost_grad = cio_planner->cost_derivative.cu;
    const Vector nabla_f = Eigen::VectorXd::Map(cost_grad.data(), cost_grad.size());
#else
    const Vector nabla_f = 2 * (X - P);
#endif

    // Compute Riemannian gradient from Euclidean one
    return project_to_tangent_space(X, nabla_f);
  };

  /// Riemannian Hessian constructor: Returns the Riemannian Hessian operator
  /// H(X): T_X(S^2) -> T_X(S^2) at X
  LinearOperatorConstructor<Vector, Vector, Vector> HC = [this, &grad_F](const Vector &X, Vector &P) {
    // Euclidean Hessian matrix
    Matrix EucHess = 2 * Matrix::Identity(DATA_SIZE, DATA_SIZE);

    // Return Riemannian Hessian-vector product operator using the
    // Euclidean Hessian
    LinearOperator<Vector, Vector, Vector> Hessian =
        [this, EucHess, &grad_F](const Vector &X, const Vector &Xdot, Vector &P) -> Vector {
      return project_to_tangent_space(X, EucHess * Xdot) - X.dot(grad_F(X, P)) * Xdot;
    };

    return Hessian;
  };

  /// Riemannian metric on S^2: this is just the usual inner-product on R^3
  RiemannianMetric<Vector, Vector, Scalar, Vector> metric =
      [](const Vector &X, const Vector &V1, const Vector &V2, const Vector &P) { return V1.dot(V2); };

  /// Projection-based retraction operator for S^2
  Retraction<Vector, Vector, Vector> retract = [](const Vector &X, const Vector &V, const Vector &P) {
    return (X + V).normalized();
  };

  /// SET INITIAL POINT

  auto *cio_planner = dynamic_cast<mjpc::CIOPlanner *>(mj_planner_);
  if (!cio_planner) {
    return;
  }

  {
    const std::vector<double> init = cio_planner->GetNominalPolicyValues(true);
    x_ = Eigen::VectorXd::Map(init.data(), init.size());
  }

  cout << "Target point: P = " << endl << P << endl << endl;
  cout << "X0 = " << endl << x_ << endl << endl;

  /// RUN GRADIENT DESCENT OPTIMIZER!

  cout << "RUNNING GRADIENT DESCENT OPTIMIZER!" << endl << endl;

  // Set gradient descent options
  GradientDescentParams<Scalar> gd_params;
  gd_params.max_iterations = 1;
  gd_params.max_ls_iterations = 1;
  gd_params.verbose = true;

  GradientDescentResult<Vector, Scalar> gd_result =
      GradientDescent<Vector, Vector, Scalar, Vector>(F, grad_F, metric, retract, x_, P, gd_params);

  cout << "Gradient descent estimate x_final:  " << endl << gd_result.x << endl << endl;

  /// RUN TNT OPTIMIZER!
  cout << "RUNNING TNT OPTIMIZER!" << endl << endl;

  // Set TNT options
  TNTParams<Scalar> tnt_params;
  tnt_params.verbose = true;

  auto tnt_result =
      TNT<Vector, Vector, Scalar, Vector>(F, grad_F, HC, metric, retract, x_, P, {}, tnt_params);
  cout << "Truncated-Newton trust-region estimate x_final:  " << endl << tnt_result.x << endl << endl;
  x_ = tnt_result.x;
}