#include "mjpc/tasks/grippers/grippers.h"

#include <string>

// MuJoCo
#include <mujoco/mujoco.h>

// Abseil
#include <absl/random/random.h>

// Mjpc
#include "mjpc/task.h"
#include "mjpc/utilities.h"

// [N] Baseline force to be applied by the gripper
static constexpr double DFT_GRIPPER_FORCE = 1;
// [N] Amplitude of the oscillations, carried out by the gripper
static constexpr double DFT_AMPLITUDE = 5;
// Duty cycle of the control signal
static constexpr double DFT_DUTY_CYCLE = 0.5;
// [s] Period of the control signal
static constexpr double DFT_PERIOD = 3;
// [m/s] Stiction tolerance
static constexpr double DFT_STICTION_TOLERANCE = 5;

// If zero, the plant is modeled as a continuous system
// If positive, the period (in seconds) of the discrete updates for the plant modeled as a discrete system
// Must be non-negative
static constexpr double DFT_MBP_DISCRETE_UPDATE_PERIOD = 5;

namespace mjpc {
std::string Grippers::XmlPath() const { return GetModelPath("grippers/task.xml"); }
std::string Grippers::Name() const { return "FreeGrippers"; }

void Grippers::ResidualFn::Residual(const mjModel* model, const mjData* data, double* residual) const {
  int counter = 0;

  // reach
  double* finger_a = SensorByName(model, data, "finger_a");
  double* box = SensorByName(model, data, "object");
  mju_sub3(residual + counter, finger_a, box);
  counter += 3;
  double* finger_b = SensorByName(model, data, "finger_b");
  mju_sub3(residual + counter, finger_b, box);
  counter += 3;

  // bring
  for (int i = 0; i < 3; i++) {
    double* object = SensorByName(model, data, std::to_string(i).c_str());
    double* target = SensorByName(model, data, (std::to_string(i) + "t").c_str());
    residual[counter++] = mju_dist3(object, target);
  }

  // control
  for (int i = 0; i < model->nu; i++) {
    residual[counter++] = data->ctrl[i];
  }

  CheckSensorDim(model, counter);
}

void Grippers::Control() {
  const Eigen::Vector2d amplitudes(DFT_AMPLITUDE, -DFT_AMPLITUDE);
  const Eigen::Vector2d duty_cycles(DFT_DUTY_CYCLE, DFT_DUTY_CYCLE);
  const Eigen::Vector2d periods(DFT_PERIOD, DFT_PERIOD);
  const Eigen::Vector2d phases = Eigen::Vector2d::Zero();
  static constexpr double gripper_force = 1;
  const auto square_force = SquareWave<2>(amplitudes, duty_cycles, periods, phases);
  const auto constant_force = Eigen::Vector2d(gripper_force, -gripper_force);
}
}  // namespace mjpc
