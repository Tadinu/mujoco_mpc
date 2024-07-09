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

#ifndef RMP_CORE_STATE_H_
#define RMP_CORE_STATE_H_

#include <Eigen/Dense>

namespace rmp {

/**
 * Represents a state with multiple derivates.
 * Currently implemented: Position + Velocity
 * @tparam d Dimensionality.
 */
template <int d>
class State {
  using Vector = Eigen::Matrix<double, d, 1>;
  using Matrix = Eigen::Matrix<double, d, d>;

 public:
  Vector pos_ = Vector::Zero();
  Matrix rot_ = Matrix::Identity();
  Vector vel_ = Vector::Zero();
  Vector size_ = Vector::Zero();
  static constexpr int dim = d;

  // more derivatives to be added later
};

}  // namespace rmp

#endif  // RMP_STATE_H
