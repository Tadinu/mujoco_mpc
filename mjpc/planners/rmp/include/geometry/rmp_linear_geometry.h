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

#ifndef RMP_GEOMETRY_LINEAR_GEOMETRY_H_
#define RMP_GEOMETRY_LINEAR_GEOMETRY_H_
#include "mjpc/planners/rmp/include/core/rmp_base_geometry.h"

namespace rmp {
/**
 * Example of a simple linear geometry, where both
 * task and configuration space are some regular
 * euclidean space R^d.
 * \tparam d Dimensionality of geometry.
 */
template <int d>
class LinearGeometry : public RMPBaseGeometry<d, d> {
public:
  // type alias for readability.
  using base = RMPBaseGeometry<d, d>;
  using VectorX = typename base::VectorX;
  using VectorQ = typename base::VectorQ;
  using StateX = typename base::StateX;
  using StateQ = typename base::StateQ;
  using J_phi = typename base::J_phi;

  /**
   * Return jacobian. As the spaces are equal, this
   * is always identity.
   */
  virtual J_phi J(const StateX&) const {
    return J_phi::Identity();
  }

  /**
   * Convert vector Q->X
   * As the spaces are equal, they are the same too.
   */
  virtual VectorX convertPosToX(const VectorQ& vector_q) const { return vector_q; }

  /**
   * Convert state Q->X
   * As the spaces are equal, they are the same too.
   */
  virtual StateX convertToX(const StateQ& state_q) const { return state_q; }

  /**
   * Convert state X->Q
   * As the spaces are equal, they are the same too.
   */
  virtual StateQ convertToQ(const StateX& state_x) const { return state_x; }

  /**
   * Convert vector X->Q
   * As the spaces are equal, they are the same too.
   */
  virtual VectorQ convertPosToQ(const VectorX& vector_x) const { return vector_x; }
};
} // namespace rmp
#endif  // RMP_GEOMETRY_LINEAR_GEOMETRY_H_
