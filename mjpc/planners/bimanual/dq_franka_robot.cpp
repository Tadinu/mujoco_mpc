#include "mjpc/planners/bimanual/dq_franka_robot.h"

// DQ
#include <dqrobotics/utils/DQ_Constants.h>

namespace DQ_robotics {
DQ_SerialManipulator DQPanda::kinematics(const std::string& name, const double r_B_O[3],
                                         const double B_Q_O[4]) {
  static const Matrix<double, 4, 7> FRANKA_DH = []() {
    Matrix<double, 4, 7> f;
    f << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // theta (q)
        0.333, 0.0, 0.316, 0.0, 0.384, 0.0,
        0.2104, // d last = flange 0.107 + EE 0.1034
        0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, // a
        0.0, -M_PI_2, M_PI_2, M_PI_2, -M_PI_2, M_PI_2, M_PI_2; // alpha
    return f;
  }();
  DQ_SerialManipulator franka(FRANKA_DH, "modified");
  const auto p = DQ(0.0, r_B_O[0], r_B_O[1], r_B_O[2]);
  auto r = DQ(B_Q_O[3], B_Q_O[0], B_Q_O[1], B_Q_O[2]);
  r = r * r.inv().norm();
  const auto base_frame = r + 0.5 * E_ * p * r;
  std::cout << "Panda " << name << " Reference Frame: " << base_frame << std::endl;
  franka.set_name(name);
  franka.set_base_frame(base_frame);
  franka.set_reference_frame(base_frame);
  return franka;
}
} // namespace DQ_robotics
