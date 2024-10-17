#pragma once

#include <mujoco/mujoco.h>

#include <memory>
#include <string>

// Eigen
#include <Eigen/Core>

#include "mjpc/task.h"

namespace mjpc {

// A simple system that outputs a square wave signal for an open loop controller.
template <int n>
class SquareWave {
public:
  using MatrixX = Eigen::Matrix<double, n, n>;
  using VectorX = Eigen::Matrix<double, n, 1>;
  // Constructs a %Square system where different amplitudes, duty cycles,
  // periods, and phases can be applied to each square wave.
  //
  // @param[in] amplitudes the square wave amplitudes. (unitless)
  // @param[in] duty_cycles the square wave duty cycles.
  //                        (ratio of pulse duration to period of the waveform)
  // @param[in] periods the square wave periods. (seconds)
  // @param[in] phases the square wave phases. (radians)
  SquareWave(const VectorX& amplitudes, const VectorX& duty_cycles, const VectorX& periods,
             const VectorX& phases)
      : amplitude_(amplitudes), duty_cycle_(duty_cycles), period_(periods), phase_(phases) {
    // Ensure the incoming vectors are all the same size.
    assert(duty_cycles.size() == amplitudes.size());
    assert(duty_cycles.size() == periods.size());
    assert(duty_cycles.size() == phases.size());
  }

private:
  std::vector<double> Values(const double time) const {
    static constexpr double M_2PI = 2 * M_PI;

    std::vector<double> output(duty_cycle_.size(), 0);
    for (int i = 0; i < duty_cycle_.size(); ++i) {
      // Add phase offset
      double t = time + (period_[i] * phase_[i] / M_2PI);

      output[i] =
          amplitude_[i] * (t - floor(t / period_[i]) * period_[i] < duty_cycle_[i] * period_[i] ? 1 : 0);
    }
  }

  const VectorX amplitude_ = VectorX::Zero();
  const VectorX duty_cycle_ = VectorX::Zero();
  const VectorX period_ = VectorX::Zero();
  const VectorX phase_ = VectorX::Zero();
};

class Grippers : public Task {
public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
  public:
    explicit ResidualFn(const Grippers* task) : mjpc::BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data, double* residual) const override;
  };
  Grippers() : residual_(this) {}

  void Control();

protected:
  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

private:
  ResidualFn residual_;
};
}  // namespace mjpc
