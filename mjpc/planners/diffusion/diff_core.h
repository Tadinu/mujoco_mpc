#pragma once

#include <vector>
#include <cmath>
#include <functional>
#include <random>

// Eigen
#include <Eigen/Dense>
#include <unsupported/Eigen/Splines>
#include <unsupported/Eigen/CXX11/Tensor>

// mjpc
#include "mjpc/threadpool.h"
#include "mjpc/utils/mjpc_math_util.h"
#include "mjpc/planners/diffusion/diff_config.h"


namespace mjpc {
class DiffusionPlanner;

class MBDPI {
public:
  DiffusionPlanner* planner_ = nullptr;
  DiffusionConfig config;
  int nu;
  Eigen::VectorXd sigmas;
  Eigen::VectorXd sigma_control;
  Eigen::VectorXd step_us;
  Eigen::VectorXd step_nodes;
  double ctrl_dt = 0.02;

  MBDPI(DiffusionPlanner* planner, const DiffusionConfig& cfg, int action_size)
    : planner_(planner), config(cfg), nu(action_size) {
    double sigma0 = 1e-2;
    double sigma1 = 1.0;
    double A = sigma0;
    double B = std::log(sigma1 / sigma0) / config.Ndiffuse;
    sigmas = Eigen::VectorXd(config.Ndiffuse);
    for (int i = 0; i < config.Ndiffuse; ++i) {
      sigmas[i] = A * std::exp(B * i);
    }

    sigma_control = Eigen::VectorXd(config.Hnode + 1);
    for (int i = 0; i <= config.Hnode; ++i) {
      sigma_control[config.Hnode - i] = std::pow(config.horizon_diffuse_factor, i);
    }

    step_us = Eigen::VectorXd::LinSpaced(config.Hsample + 1, 0, ctrl_dt * config.Hsample);
    step_nodes = Eigen::VectorXd::LinSpaced(config.Hnode + 1, 0, ctrl_dt * config.Hsample);
  }

  // Define the spline degree (e.g., cubic spline)
  static constexpr int spline_degree = 2;

  // Function to perform spline interpolation
  Eigen::MatrixXd Nodes2Us(const Eigen::MatrixXd& nodes) {
    // Number of control nodes and control dimensions
    const int num_nodes = nodes.rows();
    const int control_dim = nodes.cols();

    // Ensure that the number of nodes matches the size of step_nodes
    assert(num_nodes == step_nodes.size());

    // Normalize the step_nodes to the [0, 1] interval
    double t_min = step_nodes.minCoeff();
    double t_max = step_nodes.maxCoeff();
    Eigen::VectorXd normalized_step_nodes = (step_nodes.array() - t_min) / (t_max - t_min);

    // Normalize the step_us to the [0, 1] interval
    Eigen::VectorXd normalized_step_us = (step_us.array() - t_min) / (t_max - t_min);

    // Prepare the data for spline fitting
    // Eigen's SplineFitting expects the data in columns: each column is a point in control_dim-dimensional space
    Eigen::MatrixXd nodes_transposed = nodes.transpose();

    // Fit a spline of the specified degree
    typedef Eigen::Spline<double, Eigen::Dynamic, spline_degree> SplineType;
    SplineType spline = Eigen::SplineFitting<SplineType>::Interpolate(
        nodes_transposed, spline_degree, normalized_step_nodes);

    // Evaluate the spline at the desired points
    Eigen::MatrixXd us(control_dim, normalized_step_us.size());
    for (int i = 0; i < normalized_step_us.size(); ++i) {
      us.col(i) = spline(normalized_step_us(i));
    }

    // Transpose the result to match the original Python function's output shape
    return us.transpose();
  }

  // Function to perform spline interpolation
  Eigen::MatrixXd Us2Nodes(const Eigen::MatrixXd& us) {
    // Number of control inputs and control dimensions
    const int num_us = us.rows();
    const int control_dim = us.cols();

    // Ensure that the number of control inputs matches the size of step_us
    assert(num_us == step_us.size());

    // Normalize the step_us to the [0, 1] interval
    double t_min = step_us.minCoeff();
    double t_max = step_us.maxCoeff();
    Eigen::VectorXd normalized_step_us = (step_us.array() - t_min) / (t_max - t_min);

    // Normalize the step_nodes to the [0, 1] interval
    Eigen::VectorXd normalized_step_nodes = (step_nodes.array() - t_min) / (t_max - t_min);

    // Prepare the data for spline fitting
    // Eigen's SplineFitting expects the data in columns: each column is a point in control_dim-dimensional space
    Eigen::MatrixXd us_transposed = us.transpose();

    // Fit a spline of the specified degree
    typedef Eigen::Spline<double, Eigen::Dynamic, spline_degree> SplineType;
    SplineType spline = Eigen::SplineFitting<SplineType>::Interpolate(
        us_transposed, spline_degree, normalized_step_us);

    // Evaluate the spline at the desired node points
    Eigen::MatrixXd nodes(control_dim, normalized_step_nodes.size());
    for (int i = 0; i < normalized_step_nodes.size(); ++i) {
      nodes.col(i) = spline(normalized_step_nodes(i));
    }

    // Transpose the result to match the original Python function's output shape
    return nodes.transpose();
  }

  // Function to generate standard normal random numbers
  Eigen::Tensor<double, 3> GenerateGaussianNoise(int Nsample, int Hnode_plus1, int nu) {
    std::normal_distribution<double> dist(0.0f, 1.0f);
    Eigen::Tensor<double, 3> noise(Nsample, Hnode_plus1, nu);
    for (int i = 0; i < Nsample; ++i) {
      for (int j = 0; j < Hnode_plus1; ++j) {
        for (int k = 0; k < nu; ++k) {
          noise(i, j, k) = dist(mjpc::Random::gen);
        }
      }
    }
    return noise;
  }

  // Function to compute weighted average over the first dimension
  Eigen::MatrixXd WeightedAverage(const Eigen::MatrixXd& weights, const Eigen::Tensor<double, 3>& data) {
    const int N = data.dimension(0); // Number of samples
    const int T = data.dimension(1); // Time steps or rows
    const int D = data.dimension(2); // Features or columns

    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(T, D);

    for (int n = 0; n < N; ++n) {
      for (int t = 0; t < T; ++t) {
        for (int d = 0; d < D; ++d) {
          result(t, d) += weights(t, d) * data(n, t, d);
        }
      }
    }

    return result;
  }

  // Function to compute Softmax of a vector
  Eigen::VectorXd Softmax(const Eigen::VectorXd& x) {
    Eigen::VectorXd x_shifted = x.array() - x.maxCoeff();
    Eigen::VectorXd exp_x = x_shifted.array().exp();
    double sum_exp_x = exp_x.sum();
    return exp_x / sum_exp_x;
  }

  // Function to compute the weighted average of trajectories
  Eigen::MatrixXd SoftmaxUpdate(const Eigen::VectorXd& weights,
                                const std::vector<Eigen::MatrixXd>& Y0s) {
    int Hnode_plus1 = Y0s[0].rows();
    int nu = Y0s[0].cols();
    Eigen::MatrixXd mu_0tm1 = Eigen::MatrixXd::Zero(Hnode_plus1, nu);

    // Einstein sum
    for (size_t i = 0; i < Y0s.size(); ++i) {
      for (size_t j = 0; j < weights.size(); ++j) {
        mu_0tm1 += Y0s[i] * weights[j];
      }
    }

    return mu_0tm1;
  }

  // Main function to perform the update
  Eigen::VectorXd Estimate_mu_0tm1(const Eigen::MatrixXd& rewards) {
    // Initialize with your data
    double temp_sample = config.temp_sample; // Set your temperature scaling factor

    // Step 1: Compute the mean of the last row (rew_Ybar_i)
    double rew_Ybar_i = rewards.row(rewards.rows() - 1).mean();

    // Step 2: Compute the mean of each row (rews)
    Eigen::VectorXd rews = rewards.rowwise().mean();

    // Step 3: Compute the standard deviation of each row (std_rews)
    Eigen::VectorXd std_rews(rewards.rows());
    for (int i = 0; i < rewards.rows(); ++i) {
      Eigen::VectorXd row = rewards.row(i);
      double mean = rews(i);
      double variance = (row.array() - mean).square().sum() / (row.size() - 1);
      std_rews(i) = std::sqrt(variance);
    }

    // Step 4: Compute logp0
    Eigen::VectorXd logp0 = (rews.array() - rew_Ybar_i) / std_rews.array() / temp_sample;
    Eigen::VectorXd weights = Softmax(logp0);

    return weights;
  }

  // Function to perform the reverse diffusion step
  Eigen::MatrixXd ReverseOnce(const Eigen::MatrixXd& Ybar_i,
                              double noise_scale,
                              int horizon, ThreadPool& pool);

  // Function to perform multiple reverse diffusion steps
  Eigen::MatrixXd Reverse(const Eigen::MatrixXd& initial_nodes, int num_steps,
                          int horizon, ThreadPool& pool) {
    Eigen::MatrixXd nodes = initial_nodes;

    // Determine the number of diffusion levels
    int num_sigmas = sigmas.size();

    // Iterate over the specified number of diffusion steps
    for (int step = 0; step < num_steps; ++step) {
      // Determine the current sigma index (can be adjusted as needed)
      int sigma_index = std::min(step, num_sigmas - 1);
      double sigma = sigmas[sigma_index];

      // Apply a single reverse diffusion step
      nodes = ReverseOnce(nodes, sigma, horizon, pool);
    }

    return nodes;
  }

  // Function to shift control nodes forward by one time step
  Eigen::MatrixXd Shift(const Eigen::MatrixXd& nodes,
                        const Eigen::VectorXd& u_new) {
    // Interpolate control nodes to obtain control inputs
    Eigen::MatrixXd us = Nodes2Us(nodes);

    // Shift control inputs: remove the first row and append u_new at the end
    Eigen::MatrixXd us_shifted(us.rows(), us.cols());
    us_shifted.topRows(us.rows() - 1) = us.bottomRows(us.rows() - 1);
    us_shifted.row(us.rows() - 1) = u_new.transpose();

    // Interpolate shifted control inputs back to control nodes
    Eigen::MatrixXd nodes_shifted = Us2Nodes(us_shifted);

    return nodes_shifted;
  }

  // Function to shift control nodes forward by one time step
  Eigen::MatrixXd ShiftYFromU(const Eigen::MatrixXd& nodes,
                              const Eigen::VectorXd& u_new) {
    // Interpolate control nodes to obtain control inputs
    Eigen::MatrixXd us = Nodes2Us(nodes);

    // Shift control inputs: remove the first row and append u_new at the end
    Eigen::MatrixXd us_shifted(us.rows(), us.cols());
    us_shifted.topRows(us.rows() - 1) = us.bottomRows(us.rows() - 1);
    us_shifted.row(us.rows() - 1) = u_new.transpose();

    // Interpolate shifted control inputs back to control nodes
    Eigen::MatrixXd nodes_shifted = Us2Nodes(us_shifted);

    return nodes_shifted;
  }
};
} // namespace mjpc