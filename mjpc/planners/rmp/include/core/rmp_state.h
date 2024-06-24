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

#ifndef RMPCPP_CORE_STATE_H_
#define RMPCPP_CORE_STATE_H_

#include <Eigen/Dense>

namespace rmpcpp {

/**
 * Represents a state with multiple derivates.
 * Currently implemented: Position + Velocity
 * @tparam d Dimensionality.
 */
template <int d>
class State {
  using VectorQ = Eigen::Matrix<double, d, 1>;

 public:
  State() {}

  State(const VectorQ& pos) : pos_(pos) {}

  State(const VectorQ& pos, const VectorQ& vel) : pos_(pos), vel_(vel) {}

  VectorQ pos_ = VectorQ::Zero();
  VectorQ vel_ = VectorQ::Zero();

  // more derivatives to be added later
};

}  // namespace rmpcpp

#endif  // RMPCPP_STATE_H
