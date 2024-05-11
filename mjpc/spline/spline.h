// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MJPC_MJPC_SPLINE_SPLINE_H_
#define MJPC_MJPC_SPLINE_SPLINE_H_

#include <array>
#include <cstddef>
#include <deque>
#include <iterator>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/types/span.h>

#include <absl/log/check.h>
#include <absl/types/span.h>

namespace mjpc::spline {

enum SplineInterpolation : int {
  kZeroSpline,
  kLinearSpline,
  kCubicSpline,
};


// Represents a spline where values are interpolated based on time.
// Allows updating the spline by adding new future points, or removing old
// nodes.
// This class is not thread safe and requires locking to use.
class TimeSpline {
 public:
  explicit TimeSpline(int dim = 0,
                      SplineInterpolation interpolation = kZeroSpline,
                      int initial_capacity = 1)
      : interpolation_(interpolation), dim_(dim) {
    values_.resize(initial_capacity * dim);  // Reserve space for node values
  }

  // Copyable, Movable.
  TimeSpline(const TimeSpline& other) = default;
  TimeSpline& operator=(const TimeSpline& other) = default;
  TimeSpline(TimeSpline&& other) = default;
  TimeSpline& operator=(TimeSpline&& other) = default;

  // A view into one spline node in the spline.
  // Template parameter is needed to support both `double` and `const double`
  // views of the data.
  template <typename T>
  class NodeT {
   public:
    NodeT() : time_(0) {};
    NodeT(double time, T* values, int dim)
        : time_(time), values_(values, dim) {}

    // Copyable, Movable.
    NodeT(const NodeT& other) = default;
    NodeT& operator=(const NodeT& other) = default;
    NodeT(NodeT&& other) = default;
    NodeT& operator=(NodeT&& other) = default;

    double time() const { return time_; }

    // Returns a span pointing to the spline values of the node.
    // This function returns a non-const span, to allow spline values to be
    // modified, while the time member and underlying values pointer remain
    // constant.
    absl::Span<T> values() const { return values_; }

   private:
    double time_;
    absl::Span<T> values_;
  };

  using Node = NodeT<double>;
  using ConstNode = NodeT<const double>;

  // Iterator type for TimeSpline.
  // SplineType is TimeSpline or const TimeSpline.
  // NodeType is Node or ConstNode.
  template <typename SplineType, typename NodeType>
  class IteratorT {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename std::remove_cv_t<NodeType>;
    using difference_type = int;
    using pointer = NodeType*;
    using reference = NodeType&;

    IteratorT(SplineType* spline = nullptr, int index = 0)
        : spline_(spline), index_(index) {
      if (spline_ != nullptr && index_ != spline->Size()) {
        node_ = spline->NodeAt(index_);
      }
    }

    // Copyable, Movable.
    IteratorT(const IteratorT& other) = default;
    IteratorT& operator=(const IteratorT& other) = default;
    IteratorT(IteratorT&& other) = default;
    IteratorT& operator=(IteratorT&& other) = default;

    reference operator*() { return node_; }

    pointer operator->() { return &node_; }
    pointer operator->() const { return &node_; }

    IteratorT& operator++() {
      ++index_;
      node_ = index_ == spline_->Size() ? NodeType() : spline_->NodeAt(index_);
      return *this;
    }

    IteratorT operator++(int) {
      IteratorT tmp = *this;
      ++(*this);
      return tmp;
    }

    IteratorT& operator--() {
      --index_;
      node_ = spline_->NodeAt(index_);
      return *this;
    }

    IteratorT operator--(int) {
      IteratorT tmp = *this;
      --(*this);
      return tmp;
    }

    IteratorT& operator+=(difference_type n) {
      if (n != 0) {
        index_ += n;
        node_ =
            index_ == spline_->Size() ? NodeType() : spline_->NodeAt(index_);
      }
      return *this;
    }

    IteratorT& operator-=(difference_type n) { return *this += -n; }

    IteratorT operator+(difference_type n) const {
      IteratorT tmp(*this);
      tmp += n;
      return tmp;
    }

    IteratorT operator-(difference_type n) const {
      IteratorT tmp(*this);
      tmp -= n;
      return tmp;
    }

    friend IteratorT operator+(difference_type n, const IteratorT& it) {
      return it + n;
    }

    friend difference_type operator-(const IteratorT& x, const IteratorT& y) {
      CHECK_EQ(x.spline_, y.spline_)
          << "Comparing iterators from different splines";
      if (x != y) return (x.index_ - y.index_);
      return 0;
    }

    NodeType operator[](difference_type n) const { return *(*this + n); }

    friend bool operator==(const IteratorT& x, const IteratorT& y) {
      return x.spline_ == y.spline_ && x.index_ == y.index_;
    }

    friend bool operator!=(const IteratorT& x, const IteratorT& y) {
      return !(x == y);
    }

    friend bool operator<(const IteratorT& x, const IteratorT& y) {
      CHECK_EQ(x.spline_, y.spline_)
          << "Comparing iterators from different splines";
      return x.index_ < y.index_;
    }

    friend bool operator>(const IteratorT& x, const IteratorT& y) {
      return y < x;
    }

    friend bool operator<=(const IteratorT& x, const IteratorT& y) {
      return !(y < x);
    }

    friend bool operator>=(const IteratorT& x, const IteratorT& y) {
      return !(x < y);
    }

   private:
    SplineType* spline_ = nullptr;
    int index_ = 0;
    NodeType node_;
  };

  using iterator = IteratorT<TimeSpline, Node>;
  using const_iterator = IteratorT<const TimeSpline, ConstNode>;

  std::size_t Size() const { return times_.size(); }

  Node NodeAt(int index) {
    int values_index_ = values_begin_ + index * dim_;
    if (values_index_ >= values_.size()) {
      values_index_ -= values_.size();
      CHECK_LE(values_index_, values_.size());
    }
    return Node(times_[index], values_.data() + values_index_, dim_);
  }

  ConstNode NodeAt(int index) const {
    int values_index_ = values_begin_ + index * dim_;
    if (values_index_ >= values_.size()) {
      values_index_ -= values_.size();
      CHECK_LE(values_index_, values_.size());
    }
    return ConstNode(times_[index], values_.data() + values_index_, dim_);
  }

  iterator begin() {
    return iterator(this, 0);
  }

  iterator end() {
    return iterator(this, times_.size());
  }

  const_iterator cbegin() const {
    return const_iterator(this, 0);
  }

  const_iterator cend() const {
    return const_iterator(this, times_.size());
  }

  // Set Interpolation
  void SetInterpolation(SplineInterpolation interpolation) {
    interpolation_ = interpolation;
  }

  SplineInterpolation Interpolation() const { return interpolation_; }

  int Dim() const { return dim_; }

  // Reserves memory for at least num_nodes. If the spline already contains
  // more nodes, does nothing.
  void Reserve(int num_nodes) {
    if (num_nodes * dim_ <= values_.size()) {
      return;
    }
    if (values_begin_ < values_end_ || times_.empty()) {
      // Easy case: just resize the values_ vector and remap the spans if needed,
      // without any further data copies.
      values_.resize(num_nodes * dim_);
    } else {
      std::vector<double> new_values(num_nodes * dim_);
      // Copy all existing values to the start of the new vector
      std::copy(values_.begin() + values_begin_, values_.end(),
                new_values.begin());
      std::copy(values_.begin(), values_.begin() + values_end_,
                new_values.begin() + values_.size() - values_begin_);
      values_ = std::move(new_values);
      values_begin_ = 0;
      values_end_ = times_.size() * dim_;
    }
  }

  void Sample(double time, absl::Span<double> values) const {
    CHECK_EQ(values.size(), dim_)
        << "Tried to sample " << values.size()
        << " values, but the dimensionality of the spline is " << dim_;

    if (times_.empty()) {
      std::fill(values.begin(), values.end(), 0.0);
      return;
    }

    auto upper = std::upper_bound(times_.begin(), times_.end(), time);
    if (upper == times_.end()) {
      ConstNode n = NodeAt(upper - times_.begin() - 1);
      std::copy(n.values().begin(), n.values().end(), values.begin());
      return;
    }
    if (upper == times_.begin()) {
      ConstNode n = NodeAt(upper - times_.begin());
      std::copy(n.values().begin(), n.values().end(), values.begin());
      return;
    }

    auto lower = upper - 1;
    double t = (time - *lower) / (*upper - *lower);
    ConstNode lower_node = NodeAt(lower - times_.begin());
    ConstNode upper_node = NodeAt(upper - times_.begin());
    switch (interpolation_) {
      case SplineInterpolation::kZeroSpline:
        std::copy(lower_node.values().begin(), lower_node.values().end(),
                  values.begin());
        return;
      case SplineInterpolation::kLinearSpline:
        for (int i = 0; i < dim_; i++) {
          values[i] =
              lower_node.values().at(i) * (1 - t) + upper_node.values().at(i) * t;
        }
        return;
      case SplineInterpolation::kCubicSpline: {
        std::array<double, 4> coefficients =
            CubicCoefficients(time, lower - times_.begin());
        for (int i = 0; i < dim_; i++) {
          double p0 = lower_node.values().at(i);
          double m0 = Slope(lower - times_.begin(), i);
          double m1 = Slope(upper - times_.begin(), i);
          double p1 = upper_node.values().at(i);
          values[i] = coefficients[0] * p0 + coefficients[1] * m0 +
                      coefficients[2] * p1 + coefficients[3] * m1;
        }
        return;
      }
      default:
        CHECK(false) << "Unknown interpolation: " << interpolation_;
    }
  }

  std::vector<double> Sample(double time) const {
    std::vector<double> values(dim_);
    Sample(time, absl::MakeSpan(values));
    return values;
  }

  int DiscardBefore(double time) {
    // Find the first node that has n.time > time.
    auto last_node = std::upper_bound(times_.begin(), times_.end(), time);
    if (last_node == times_.begin()) {
      return 0;
    }

    // If using cubic interpolation, include not just the last node before `time`,
    // but the one before that.
    int keep_nodes = interpolation_ == SplineInterpolation::kCubicSpline ? 1 : 0;
    last_node--;
    while (last_node != times_.begin() && keep_nodes) {
      last_node--;
      keep_nodes--;
    }
    int nodes_to_remove = last_node - times_.begin();

    times_.erase(times_.begin(), last_node);
    values_begin_ += dim_ * nodes_to_remove;
    if (values_begin_ >= values_.size()) {
      values_begin_ -= values_.size();
      CHECK_LE(values_begin_, values_.size());
    }
    return nodes_to_remove;
  }

  void Clear() {
    times_.clear();
    values_begin_ = 0;
    values_end_ = 0;
    // Don't change capacity_ or reset values_.
  }

  // Adds a new set of values at the given time. Implementation is only
  // efficient if time is later than any previously added nodes.
  Node AddNode(double time) {
    return AddNode(time, absl::Span<const double>());  // Default empty values
  }

  Node AddNode(double time,
                                      absl::Span<const double> new_values) {
    CHECK(new_values.size() == dim_ || new_values.empty());
    // TODO(nimrod): Implement node insertion in the middle of the spline
    CHECK(times_.empty() || time > times_.back() || time < times_.front())
        << "Adding nodes to the middle of the spline isn't supported.";
    if (times_.size() * dim_ >= values_.size()) {
      Reserve(times_.size() * 2);
    }
    Node new_node;
    if (times_.empty() || time > times_.back()) {
      CHECK_LE(values_end_ + dim_, values_.size());
      times_.push_back(time);
      values_end_ += dim_;
      if (values_end_ >= values_.size()) {
        CHECK_EQ(values_end_, values_.size());
        values_end_ -= values_.size();
      }
      new_node = NodeAt(times_.size() - 1);
    } else {
      CHECK_LT(time, times_.front());
      values_begin_ -= dim_;
      if (values_begin_ < 0) {
        values_begin_ += values_.size();
      }
      CHECK_LE(values_begin_ + dim_, values_.size());
      times_.push_front(time);
      new_node = NodeAt(0);
    }
    if (!new_values.empty()) {
      std::copy(new_values.begin(), new_values.end(), new_node.values().begin());
    } else {
      std::fill(new_node.values().begin(), new_node.values().end(), 0.0);
    }
    return new_node;
  }

  std::array<double, 4> CubicCoefficients(
      double time, int lower_node_index) const {
    std::array<double, 4> coefficients;
    int upper_node_index = lower_node_index + 1;
    CHECK(upper_node_index != times_.size())
        << "CubicCoefficients shouldn't be called for boundary conditions.";
    double lower = times_[lower_node_index];
    double upper = times_[upper_node_index];
    double t = (time - lower) / (upper - lower);

    coefficients[0] = 2.0 * t*t*t - 3.0 * t*t + 1.0;
    coefficients[1] =
        (t*t*t - 2.0 * t*t + t) * (upper - lower);
    coefficients[2] = -2.0 * t*t*t + 3 * t*t;
    coefficients[3] = (t*t*t - t*t) * (upper - lower);

    return coefficients;
  }

  double Slope(int node_index, int value_index) const {
    ConstNode node = NodeAt(node_index);
    if (node_index == 0) {
      ConstNode next = NodeAt(node_index + 1);
      // one-sided finite-diff
      return (next.values().at(value_index) - node.values().at(value_index)) /
            (next.time() - node.time());
    }
    ConstNode prev = NodeAt(node_index - 1);
    if (node_index == times_.size() - 1) {
      return (node.values().at(value_index) - prev.values().at(value_index)) /
            (node.time() - prev.time());
    }
    ConstNode next = NodeAt(node_index + 1);
    return 0.5 * (next.values().at(value_index) - node.values().at(value_index)) /
              (next.time() - node.time()) +
          0.5 * (node.values().at(value_index) - prev.values().at(value_index)) /
              (node.time() - prev.time());
  }

 private:
  SplineInterpolation interpolation_;

  int dim_;

  // The time values for each node. This is kept sorted.
  std::deque<double> times_;

  // The raw node values. Stored in a ring buffer, which is resized whenever
  // too many nodes are added.
  std::vector<double> values_;

  // The index in values_ for the data of the earliest node.
  int values_begin_ = 0;

  // One past the index in values_ for the end of the data of the last node.
  // If values_end_ == values_begin_, either there's no data (nodes_ is empty),
  // or the values_ buffer is full.
  int values_end_ = 0;
};

}  // namespace mjpc::spline

#endif  // MJPC_MJPC_SPLINE_SPLINE_H_
