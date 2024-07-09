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

#ifndef RMP_UTIL_H_
#define RMP_UTIL_H_

#include "mjpc/planners/fabrics/include/fab_diff_map.h"

namespace rmp {
template <int dim, typename T = double, typename TVector = Eigen::Matrix<T, dim, 1>>
TVector vectorFromScalarArray(const T* scalarArray) {
  TVector v;
  memcpy(v.data(), scalarArray, sizeof(T) * dim);
  return v;
}

template <int dim, typename T = double, typename TMatrix = Eigen::Matrix<T, dim, dim>>
TMatrix matrixFromScalarArray(const T* matrix) {
  return TMatrix(matrix[0], matrix[1], matrix[2], matrix[3], matrix[4], matrix[5], matrix[6], matrix[7],
                 matrix[8]);
}

template <int dim, typename T = double, typename TQuat = Eigen::Quaternion<T>>
TQuat quatFromScalarArray(const T* scalarQuat) {
  return TQuat(scalarQuat[0], scalarQuat[1], scalarQuat[2], scalarQuat[3]);
}

/**
 * Softmax helper function / Directionally stretched metrics
 * https://arxiv.org/abs/1801.02854
 *
 * gamma: norm of a given vector
 * alpha: weighing factor for the softmax
 */
inline double softmax(const double gamma, const double alpha) {
  return (gamma + (1.0 / alpha) * log(1 + exp(-2 * alpha * gamma)));
}

/**
 * Range iterator for multi-dimensional vectors.
 * By supplying a start vector, an end vector and and increment vector,
 * this returns an iterator that goes steps through the resulting regular grid.
 *
 * @tparam TVector Inherited from Eigen::Vectord. Defines size.
 */
template <class TVector>
class VectorRange {
public:
  /**
   * Internal Subclass that defines the
   * actual iterator that is being returned.
   * Note that the current position is no the actual numeric value
   * of that position, but an integer cardinal number that corresponds
   * to the current id of the "grid" location.
   *
   * Example:
   *  start is [-1, -1]
   *  end is [1, 1,]
   *  inc is [0.5, 0.5]
   *
   *  so a tick value of [1,2] corresponds to the number
   *  start + [1 *0.5, 2*0.5] = -0.5, 0.0
   *
   *  This is done to avoid rounding errors by stacking many float numbers
   *  and to have a clear upfront number of elements that the iterator
   *  is going through.
   */
  class ConstVectorRangeIterator {
  public:
    typedef ConstVectorRangeIterator self_type;
    typedef TVector value_type;
    typedef std::forward_iterator_tag iterator_category;

    ConstVectorRangeIterator(const VectorRange<TVector>* parent, const TVector& current)
        : current_(current), parent_(parent) {}

    self_type operator++() {
      self_type i = *this;
      current_ = parent_->getNext(current_);
      return i;
    }

    self_type operator++(int junk) {
      current_ = parent_->getNext(current_);
      return *this;
    }

    const value_type operator*() { return parent_->getValue(current_); }

    bool operator==(const self_type& rhs) { return current_ == rhs.current_ && parent_ == rhs.parent_; }

    bool operator!=(const self_type& rhs) { return current_ != rhs.current_ || parent_ != rhs.parent_; }

  private:
    TVector current_;
    const VectorRange<TVector>* parent_;
  };

  /**
   * Constructor that fully parametrizes the iterator.
   * @param start Start position (sort of "left upper corner" of a grid)
   * @param end  End position (sort of the "right lower corner" of a grid)
   * @param inc  Increments for each dimension.
   */
  VectorRange(const TVector& start, const TVector& end, const TVector& inc)
      : start_range_(start), end_range_(end), inc_(inc) {
    // calculates how many individual increments in each dimension happen to reach
    // the end
    ticks_ = Vector(((end_range_ - start_range_).array() / inc_.array()).floor());
    ticks_ = (ticks_.array() < 0).select(0, ticks_);          // set all elements smaller 0 to 0.
    ticks_ = (!ticks_.array().isFinite()).select(0, ticks_);  // set all elements NAN to 0.

    ticks_plus_one_ = ticks_;

    // Artificial end position, as the convention for iterators is to return
    // last+1  as their end.
    ticks_plus_one_[0] += 1;
  }

  /**
   * Calculates the length of the iterator.
   *  Assumes that the END of the range is included too.
   *  todo(mpantic): Logic is a bit convoluted here, maybe there's a nicer way.
   */
  int length() {
    // return multiplicative digital root :-) (Querprodukt)
    int product = 1;
    for (int i = 0; i < ticks_.size(); ++i) {
      product *= (ticks_[i] == 0.0 ? 1.0 : ticks_[i] + 1.0);  // use 1 for values that have a zero.
    }
    return product;
  };

  /**
   * Returns the first iterator element.
   * Corresponds to all ticks in the vector to be zero.
   */
  ConstVectorRangeIterator begin() const { return ConstVectorRangeIterator(this, TVector::Zero()); }

  /**
   * Returns the end marker (last element + 1)
   */
  ConstVectorRangeIterator end() const { return ConstVectorRangeIterator(this, ticks_plus_one_); }

private:
  /**
   * Converts the current position of the loop (ticks)
   * to the value this represents.
   */
  TVector getValue(TVector& ticks) const { return start_range_ + (ticks.array() * inc_.array()).matrix(); }

  /**
   *  Returns the next tick vector based on a current position.
   */
  TVector getNext(TVector& current_ticks) const {
    TVector next(current_ticks);
    next[0]++;

    uint i = 0;
    while (next[i] > ticks_[i]) {
      next[i] = 0;
      i++;
      if (i < current_ticks.rows()) {
        next[i]++;
      } else {
        return ticks_plus_one_;
      }
    }
    return next;
  }

  const TVector start_range_, end_range_, inc_;
  TVector ticks_;  ///< integer ticks to increment through
  TVector ticks_plus_one_;
};
}  // namespace rmp

#endif  // RMP_UTIL_H_
