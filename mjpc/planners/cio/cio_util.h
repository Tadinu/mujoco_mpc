#pragma once

#include <Eigen/Core>
class CIOUtils {
public:
  static Eigen::Vector3d calc_derivative(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, double delta) {
    return (p1 - p2) / delta;
  }
};
