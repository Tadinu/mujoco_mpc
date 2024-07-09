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

#ifndef RMP_GEOMETRY_CYLINDRICAL_GEOMETRY_H_
#define RMP_GEOMETRY_CYLINDRICAL_GEOMETRY_H_

#include "mjpc/planners/rmp/include/core/rmp_base_geometry.h"

namespace rmp {

/**
 * Example of a cylindrical geometry that maps to a plane.
 * Here, the task space is the unit sphere, and the configuration space is R^3
 *
 * X Task space coordinates are: theta, rho, z
 * Q Configuration space coordinates are: x,y,z
 */
class CylindricalGeometry : public RMPBaseGeometry<3, 3> {
public:
  // type alias for readability.
  using base = RMPBaseGeometry<3, 3>;
  using VectorX = base::VectorX;
  using Vector = base::VectorQ;
  using StateX = base::StateX;
  using StateQ = base::StateQ;
  using J_phi = base::J_phi;

 protected:
  /**
   * Return jacobian.
   */
  virtual J_phi J(const StateX &state_x) const {
    J_phi mtx_j(J_phi::Identity());

    //base::Vector q;

    mtx_j(0, 0) = cos(state_x.pos_.y());
    mtx_j(0, 1) = sin(state_x.pos_.y());
    mtx_j(1, 0) = -state_x.pos_.x() * sin(state_x.pos_.y());
    mtx_j(1, 1) = state_x.pos_.x() * cos(state_x.pos_.y());
    // Rest is taken care of by the identity initializer.
    return mtx_j;
  }

 public:
  virtual VectorX convertPosToX(const VectorQ &pos_q) const {
    // Standard formula for cylindrical coordinates
    VectorX pos_x;

    // rho
    pos_x.x() = pos_q.template topRows<2>().norm();  // sqrt(x^2+y^2)

    // theta
    pos_x.y() = atan2(pos_q.y(), pos_q.x());

    // z
    pos_x.z() = pos_q.z();

    return pos_x;
  }

  virtual StateX convertToX(const StateQ &state_q) const {
    // Standard formula for cylindrical coordinates
    StateX state_x;

    state_x.pos_ = convertPosToX(state_q.pos_);

    // J(state_x) only works because we know J only uses position...
    state_x.vel_ = J(state_x) * state_q.vel_;

    return state_x;
  }

  virtual VectorQ convertPosToQ(const VectorX &vector_x) const {
    // Standard formula for cylindrical coordinates
    VectorQ vector_q;

    // rho
    vector_q.x() = vector_x.x() * cos(vector_x.y());

    // theta
    vector_q.y() = vector_x.x() * sin(vector_x.y());

    // z
    vector_q.z() = vector_x.z();
    return vector_q;
  }

  virtual StateQ convertToQ(const StateX &state_x) const {
    // Standard formula for cylindrical coordinates
    StateQ state_q;
    state_q.pos_ = convertPosToQ(state_x.pos_);
    state_q.vel_ = J(state_q) * state_x.vel_;

    return state_q;
  }
};
}  // namespace rmp
#endif  // RMP_GEOMETRY_CYLINDRICAL_GEOMETRY_H_
