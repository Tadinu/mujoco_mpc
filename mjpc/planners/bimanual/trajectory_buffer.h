#pragma once

#include <array>
#include <vector>

#include <Eigen/Dense>

namespace mjpc {
class TrajectoryBuffer {
public:
  TrajectoryBuffer() = delete;
  explicit TrajectoryBuffer(int size);
  Eigen::Vector3d& operator[](int i);
  void clear();
  bool empty() const;
  const Eigen::Vector3d& end();
  bool full() const;
  Eigen::Vector3d get();
  int max_size() const;
  bool put(const Eigen::Vector3d& qD);
  int size() const;

private:
  int head_ = 0, tail_ = 0;
  std::vector<Eigen::Vector3d> buf_;
  bool full_ = false;
  const int size_;
};
}
