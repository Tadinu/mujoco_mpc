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

#ifndef RMP_BASE_GEOMETRY_BASE_H_
#define RMP_BASE_GEOMETRY_BASE_H_

#include "mjpc/planners/rmp/include/core/rmp_state.h"
#include "mjpc/planners/rmp/include/policies/rmp_base_policy.h"
#include "mjpc/planners/rmp/include/policies/rmp_policy_value.h"

#include <Eigen/Dense>

namespace rmp {

/** Abstract Base class that fully defines a geometry mapping between two spaces
 * Q and X. It mostly serves to pull everything together and ensure the correct
 * sizes of all used vectors.
 *
 * Q is generally used as configuration space, and X as task space.
 *
 *  \tparam k Dimensionality of Task TSpace \f$ \mathcal{X} \f$
 *  \tparam d Dimensionality of Config TSpace \f$ \mathcal{Q} \f$
 *
 */
template <int k, int d>
class RMPBaseGeometry {
 public:
  /// Type alias for a Vector belonging to task Space X
  using MatrixX = Eigen::Matrix<double, k, k>;
  using VectorX = Eigen::Matrix<double, k, 1>;
  using PolicyX = PolicyValue<k>;
  using StateX = State<k>;

  /// Type alias for a Vector belonging to Config Space Q
  using MatrixQ = Eigen::Matrix<double, d, d>;
  using VectorQ = Eigen::Matrix<double, d, 1>;
  using PolicyQ = PolicyValue<d>;
  using StateQ = State<d>;

  /// Type for J as defined in Eq 1 in [1]
  using J_phi = Eigen::Matrix<double, k, d>;

  /*
   * Class that represents a fully parametrized geometry
   * that allows pushing and polling.
   * Internal class, as it is built by RMPBaseGeometry only
   */
  class ParametrizedGeometry {
    // Corresponding Jacobian at this state
    J_phi J_;

    // Position/Velocity of agent at which this geometry is valid
    StateX stateX_;

    // Obstacle states
    std::vector<StateX> obstacle_statesX_;

   public:
    ParametrizedGeometry(const J_phi& J, const StateX& agent_state,
                         const std::vector<StateX>& obstacle_states) :
      J_(J), stateX_(agent_state), obstacle_statesX_(obstacle_states) {}

    /**
     * Evaluates a given policy and then pulls it.
     * @param policy_noneval Non-evaluated policy to pull
     */
    template <class TNormSpace>
    PolicyQ pull(RMPPolicyBase<TNormSpace>* noneval_policyX) {
      return pull(noneval_policyX->evaluateAt(stateX_, obstacle_statesX_));
    }

    /**
     * Takes an evaluated policy and pulls it
     * @param policy Evaluated policy
     */
    PolicyQ pull(const PolicyX& policy) {
      // RMP: f: instantaneous acceleration, A: Riemannian metric as the weight of the policy
      // https://arxiv.org/pdf/1801.02854 - Eq 10, 11
      MatrixQ A = J_.transpose() * policy.A_ * J_; // Pullback metric
      VectorQ f = PolicyX::pinv(A) * J_.transpose() * policy.A_ * policy.f_;
      return {f, A};
    }

    /**
     * Takes an evaluated policy and pushes it
     * @param policy Evaluated Policy
     */
    PolicyX push(const PolicyQ policy) {
      // https://arxiv.org/pdf/1801.02854 - Eq 12
      // todo(mpantic): Check if this correct. I think its currently not used.
      MatrixX B = J_.transpose() * policy.A_ * J_; // Pushforward metric
      auto J_pinv = PolicyX::pinv(J_);
      MatrixX A = J_pinv.transpose() * B * J_pinv;
      VectorX f = J_ * policy.f_;
      return {f, A};
    }
  };

  /// Static accessor for dimension K of X
  static constexpr int K = k;

  /// Static accessor for dimension D of Q
  static constexpr int D = d;

  /**
   * Create a fully parametrized geometry for the agent & obstacles
   * @param state Agent state to parametrize
   * @param obstacle_states States of obstacles to parametrize
   */
  ParametrizedGeometry createParametrized(const StateX& agent_state, const std::vector<StateX>& obstacle_states) {
    return ParametrizedGeometry(J(agent_state), agent_state, obstacle_states);
  }

  /**
   * To be implemented in derived classes.
   * Performs position from Q to X.
   */
  virtual VectorX convertPosToX(const VectorQ &vector_q) const = 0;

  /**
   * To be implemented in derived classes.
   * Performs coordinate conversion from Q to X.
   */
  virtual StateX convertToX(const StateQ &state_q) const = 0;

  /**
   * To be implemented in derived classes.
   * Performs coordinate conversion from X to Q.
   */
  virtual StateQ convertToQ(const StateX &state_x) const = 0;

  /**
   * To be implemented in derived classes.
   * Performs position conversion from X to Q.
   */
  virtual VectorQ convertPosToQ(const VectorX &vector_x) const = 0;

  /**
   * To be implemented in derived classes.
   * Calculates Jacobian at position x/x_dot.
   */
  virtual J_phi J(const StateX&) const = 0;
};
}  // namespace rmp

#endif  // RMP_BASE_GEOMETRY_BASE_H_
