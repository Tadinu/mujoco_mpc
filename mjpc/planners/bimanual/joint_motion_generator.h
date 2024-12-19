// Copyright (c) 2020 Lehrstuhl für Robotik und Sysstemintelligenz, TU München
#pragma once

#include <Eigen/Dense>

typedef Eigen::Matrix<double, 14, 1> Vector14d;

class JointMotionGenerator {
public:
  static void generateC1Trajectory(Vector14d& q_d,
                                   Vector14d& qD_d, double v, const Vector14d& q,
                                   const Vector14d& goal);
};
