#pragma once

// Eigen
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>

#include "mjpc/planners/cio/cio_common.h"

class CIOUtils {
public:
  static Eigen::Vector3d calc_derivative(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, double delta) {
    return (p1 - p2) / delta;
  }

  // Ref: https://gist.github.com/lorenzoriano/5414671
  template <typename T>
  static std::vector<T> linspace(double start, double end, int num) {
    std::vector<T> linspaced;

    if (0 != num) {
      if (1 == num) {
        linspaced.push_back(static_cast<T>(start));
      } else {
        double delta = (end - start) / (num - 1);

        for (auto i = 0; i < (num - 1); ++i) {
          linspaced.push_back(static_cast<T>(start + delta * i));
        }
        // ensure that start and end are exactly the same as the input
        linspaced.push_back(static_cast<T>(end));
      }
    }
    return linspaced;
  }

  static std::vector<Eigen::Vector3d> linspace_vectors(const Eigen::Vector3d& vec0,
                                                       const Eigen::Vector3d& vec1, int num_steps) {
    std::vector<Eigen::Vector3d> out_vec;
    std::vector<std::vector<double>> linspaced(num_steps, {0, 0, 0});
    for (auto j = 0; j < 3; ++j) {
      const auto left = vec0[j];
      const auto right = vec1[j];
      linspaced[j] = linspace<double>(left, right, num_steps);
    }
    for (auto i = 0; i < num_steps; ++i) {
      out_vec.emplace_back(linspaced[0][i], linspaced[1][i], linspaced[2][i]);
    }
    return out_vec;
  }

  template <typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>>
  static void add_gaussian_noise(T& in) {
    // perturb all vars by gaussian noise
    static constexpr T mean = 0.;
    static constexpr T var = 0.01;

    static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::default_random_engine generator(seed);
    static std::normal_distribution<T> distribution(mean, var);

    in += std::abs(distribution(generator));
  }

  // Function to create spline functions
  static std::vector<Eigen::Spline<double, 1>> create_splines(const std::vector<Eigen::Vector3d>& pos_traj_K,
                                                              const std::vector<Eigen::Vector3d>& vel_traj_K,
                                                              const CIOConfig& config) {
    std::vector<Eigen::Spline<double, 1>> splines;
    const auto x = Eigen::Vector3d::LinSpaced(config.K + 1, 0.0, config.T_final());

    for (int dim = 0; dim < 3; ++dim) {
      Eigen::MatrixXd y(config.K + 1, 2);  // Initialize y matrix
      for (int k = 0; k <= config.K; ++k) {
        y(k, 0) = pos_traj_K[k](dim);  // Position
        y(k, 1) = vel_traj_K[k](dim);  // Velocity
      }

      // Create spline from derivatives
      Eigen::Spline<double, 1> spline =
          Eigen::SplineFitting<Eigen::Spline<double, 1>>::Interpolate(y.col(0), 3, x);
      splines.push_back(spline);
    }
    return splines;
  }
};
