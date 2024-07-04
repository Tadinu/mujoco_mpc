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

#ifndef RMP_POLICIES_SIMPLE_TARGET_POLICY_H_
#define RMP_POLICIES_SIMPLE_TARGET_POLICY_H_

#include "rmp_base_policy.h"

namespace rmp {
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

  SimpleTargetPolicy() = default;
  /**
   * Sets up the policy.
   * target is the target to move to.
   * A is the metric to be used.
   * alpha, beta and c are tuning parameters.
   */
  SimpleTargetPolicy(const Vector& target, const Matrix& A,
                     double kp, double kd, double alpha)
    : target_(target), kp_(kp), kd_(kd), alpha_(alpha) {
    this->A_static_ = A;
  }

  explicit SimpleTargetPolicy(const Vector& target) : target_(target) {
  }

  void operator()(const Vector& target, const Matrix& A, double kp, double kd,
                  double alpha) {
    this->target_ = target;
    this->kp_ = kp;
    this->kd_ = kd;
    this->alpha_ = alpha;
    this->A_static_ = A;
  }

  PValue evaluateAt(const PState& state, const std::vector<PState>&) override {
    Vector f = kp_ * this->soft_norm(this->space_.minus(target_, state.pos_), alpha_) -
               kd_ * state.vel_;
    return {std::move(f), this->A_static_};
  }

  void updateParams(const double kp, const double kd, const double alpha) {
    kp_ = kp;
    kd_ = kd;
    alpha_ = alpha;
  }

public:
  Vector target_ = Vector::Zero();
  double kp_{1.0}, kd_{8.0}, alpha_{0.005};
};
} // namespace rmp

#endif  // RMP_POLICIES_SIMPLE_TARGET_POLICY_H_
