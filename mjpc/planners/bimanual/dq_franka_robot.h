#pragma once

#include <dqrobotics/robot_modeling/DQ_SerialManipulator.h>

namespace DQ_robotics {
class DQPanda {
public:
  static DQ_SerialManipulator kinematics(const std::string& name, const double r_B_O[3],
                                         const double B_Q_O[4]);
};
} // namespace DQ_robotics
