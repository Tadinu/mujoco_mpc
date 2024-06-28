/**
 * This file is part of RMPCPP
 *
 * Copyright (C) 2020 Michael Pantic <mpantic at ethz dot ch>
 *
 * RMPCPP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * RMPCPP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received SenRMPSimpleTargetPolicya copy of the GNU General Public License
 * along with RMPCPP. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef RMPCPP_POLICIES_SIMPLE_TARGET_POLICY_H_
#define RMPCPP_POLICIES_SIMPLE_TARGET_POLICY_H_

#include "mjpc/planners/rmp/include/core/rmp_base_policy.h"

namespace rmpcpp {

/**
 * Defines a simple dimensional target policy, as described in [1].
 */
template <class TNormSpace>
class SimpleTargetPolicy : public RMPPolicyBase<TNormSpace> {

 public:
  using Vector = typename RMPPolicyBase<TNormSpace>::Vector;
  using Matrix = typename RMPPolicyBase<TNormSpace>::Matrix;
  using PValue = typename RMPPolicyBase<TNormSpace>::PValue;
  using PState = typename RMPPolicyBase<TNormSpace>::PState;

 SimpleTargetPolicy() {}
  /**
   * Sets up the policy.
   * target is the target to move to.
   * A is the metric to be used.
   * alpha, beta and c are tuning parameters.
   */
  SimpleTargetPolicy(const Vector& target, const Matrix& A,
                     double alpha, double beta, double c)
      : target_(target), alpha_(alpha), beta_(beta), c_(c) {
    this->A_static_ = A;
  }

  SimpleTargetPolicy(const Vector& target) : target_(target) {}

  void operator()(const Vector& target, const Matrix& A, double alpha, double beta,
                  double c) {
    this->target_ = target;
    this->alpha_ = alpha;
    this->beta_ = beta;
    this->c_ = c;
    this->A_static_ = A;
  }

  virtual PValue evaluateAt(const PState &state, const std::vector<PState>&) {
    Vector f = alpha_ * soft_norm(this->space_.minus(target_, state.pos_)) -
                beta_ * state.vel_;
    return {f, this->A_static_};
  }

  void updateParams(double alpha, double beta, double c){
    alpha_ = alpha;
    beta_ = beta;
    c_ = c;
  }

public:
  /**
   *  Normalization helper function.
   */
  inline Vector soft_norm(const Vector& v) { return v / h(this->space_.norm(v)); }

  /**
   * Softmax helper function
   */
  inline double h(const double z) {
    return (z + c_ * log(1 + exp(-2 * c_ * z)));
  }

  Vector target_ = Vector::Zero();
  double alpha_{1.0}, beta_{8.0}, c_{0.005};
};

}  // namespace rmpcpp

#endif  // RMPCPP_POLICIES_SIMPLE_TARGET_POLICY_H_
