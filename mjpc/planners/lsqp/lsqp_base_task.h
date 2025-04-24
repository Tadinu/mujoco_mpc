#pragma once
#include <stdexcept>

#include <Eigen/Core>

// absl
#include <absl/status/status.h>

// mjpc
#include "mjpc/planners/lsqp/lsqp_core_task.h"
#include "mjpc/planners/lsqp/lsqp_config.h"

namespace mjpc {
// Quadratic objective of the form :0.5 *x^T * H * x + c^T *x
struct LsqpObjective {
  // Hessian matrix, of shape (nv, nv)
  Eigen::MatrixXd H;
  // Linear vector of shape (nv)
  Eigen::VectorXd c;

  double Value(const Eigen::VectorXd& x) const {
    // Returns the value of the objective at the input vector
    return x.transpose() * H * x + c.dot(x);
  }

  void PrintSelf() const {
    mjpc::print("H", H);
    mjpc::print("c", c.transpose());
  }
};

class LsqpBaseTask : public LsqpCoreTask {
public:
  LsqpBaseTask() = default;

  LsqpBaseTask(std::string name, const mjModel* model, Eigen::VectorXd cost, double gain,
               double lm_damping) :
    name_(std::move(name)),
    cost_(std::move(cost)), gain_(gain), lm_damping_(lm_damping), nq_(model->nq) {
    if (gain < 0.0 || gain > 1.0) {
      throw std::invalid_argument("`gain` must be in the range [0, 1]");
    }

    if (lm_damping < 0.0) {
      throw std::invalid_argument("`lm_damping` must be >= 0");
    }
  }

  ~LsqpBaseTask() override = default;

  std::string Name() const { return name_; }
  virtual bool Empty() const { return false; }
  virtual Eigen::VectorXd ComputeError(const mjData* data, const LsqpConfig& config) const = 0;
  virtual Eigen::MatrixXd ComputeJac(const mjData* data, const LsqpConfig& config) const = 0;

  LsqpObjective ComputeQPObjective(const mjData* data, const LsqpConfig& config) const {
    const int ndofs = config.ndofs();
    Eigen::MatrixXd jac = ComputeJac(data, config);
    //mjpc::print(jac);
    const int jac_rows = jac.rows();
    const int jac_cols = jac.cols();
    const bool bTrim_dofs = config.MjModel()->nv > ndofs;
    if (bTrim_dofs) {
      jac = jac.block(0, 0, is_frame_task_ ? jac_rows : std::min(jac_rows, ndofs), std::min(jac_cols, ndofs));
    }
    Eigen::VectorXd minus_gain_error = -gain_ * ComputeError(data, config); // (k,)

    if (bTrim_dofs && (!is_frame_task_)) {
      minus_gain_error = minus_gain_error.head(ndofs);
    }
    Eigen::MatrixXd weight = cost_.asDiagonal();
    if (bTrim_dofs && (!is_frame_task_)) {
      weight = weight.block(0, 0, ndofs, ndofs);
    }

    const Eigen::MatrixXd weighted_jacobian = weight * jac;
    const Eigen::VectorXd weighted_error = weight * minus_gain_error;

    const double mu = lm_damping_ * weighted_error.dot(weighted_error);
    const Eigen::MatrixXd eye_tg = Eigen::MatrixXd::Identity(ndofs, ndofs);

    Eigen::MatrixXd H = weighted_jacobian.transpose() * weighted_jacobian + mu * eye_tg; //(ndofs, ndofs)
    Eigen::VectorXd c = -weighted_error.transpose() * weighted_jacobian; // (ndofs,)
    return LsqpObjective{.H = std::move(H), .c = std::move(c)};
  }

protected:
  std::string name_;
  int nq_ = 0; // model->nq
  int k_ = 0; // in [1, model->nv]
  Eigen::VectorXd cost_;
  double gain_ = 1.0;
  double lm_damping_ = 1.0;
  bool is_frame_task_ = false;
};
} // end namespace mjpc