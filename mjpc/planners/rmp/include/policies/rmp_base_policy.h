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
 * You should have received a copy of the GNU General Public License
 * along with RMPCPP. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef RMPCPP_CORE_POLICY_BASE_H_
#define RMPCPP_CORE_POLICY_BASE_H_

#include <Eigen/Dense>
#include <Eigen/QR>
#include <iostream>
#include <memory>

// MUJOCO
#include "mujoco/mjvisualize.h"

// MJPC
#include "mjpc/planners/rmp/include/core/rmp_parameters.h"
#include "mjpc/planners/rmp/include/core/rmp_space.h"
#include "mjpc/planners/rmp/include/core/rmp_state.h"
#include "rmp_policy_value.h"

// Macro to mark unused variables s.t. no unused warning appears.
#define ACK_UNUSED(expr) \
  do {                   \
    (void)(expr);        \
  } while (0)

namespace rmpcpp {

/**
 * Holds a plain basic n-dimensional Riemannian Motion Policy
 * as defined in [1], Chapter IV.
 *
 * Implements all mathematical base operations on RMPs.
 *
 * \tparam TNormSpace TSpace in which the policy is active (defines dimensionality
 * and distance norm)
 */
template <class TNormSpace>
class RMPPolicyBase {
 public:
  static constexpr int d = TNormSpace::dim;
  using Matrix = Eigen::Matrix<double, TNormSpace::dim, TNormSpace::dim>;
  using Vector = Eigen::Matrix<double, TNormSpace::dim, 1>;
  using PValue = PolicyValue<TNormSpace::dim>;
  using PState = State<TNormSpace::dim>;

  virtual ~RMPPolicyBase() = default;

  // to be implemented in derivatives.
  virtual PValue evaluateAt(const PState& /*agent_state*/, const std::vector<PState>& /*obstacle_states*/) = 0;
  /** Asynchronous start of evaluation. If implemented will make a subsequent (blocking) call to evaluateAt
   * (with the same state) faster. */
  virtual void startEvaluateAsync(const PState&, const std::vector<PState>& obstacle_states){}; // As a default derivatives don't have to implement this
  virtual void abortEvaluateAsync(){};

  template<typename TPolicy, const ERMPPolicyType policy_type,
           typename TPolicyConfigs = std::conditional_t<policy_type == RAYCASTING,
                                                        RaycastingPolicyConfigs, ESDFPolicyConfigs>>
  static std::shared_ptr<TPolicy> MakePolicy() {
    const auto policy_configs = std::dynamic_pointer_cast<TPolicyConfigs>(RMPConfigs(policy_type).policyConfigs);
    return policy_configs ? std::make_shared<TPolicy>(*policy_configs) : nullptr;
  }

  /**
   * Setter for metric A
   * @param A Metric
   */
  inline void setA(Matrix A) { A_static_ = A; }

 public:
  RMPPolicyBase() = default;

  mjvScene* scene_ = nullptr;
  TNormSpace space_;
  Matrix A_static_ = Matrix::Identity();

  struct RayTrace {
   Vector ray_start;
   Vector ray_end;
   double distance = 0.;
  };
  std::vector<RayTrace> raytraces_;
};
}  // namespace rmpcpp

#endif  // RMPCPP_CORE_POLICY_BASE_H_
