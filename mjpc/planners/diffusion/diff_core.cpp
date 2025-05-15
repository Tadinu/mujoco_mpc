#include "mjpc/planners/diffusion/diff_core.h"
#include "mjpc/planners/diffusion/diff_planner.h"

namespace mjpc {
Eigen::MatrixXd MBDPI::ReverseOnce(const Eigen::MatrixXd& Ybar_i,
                                   double noise_scale,
                                   int horizon,
                                   ThreadPool& pool) {
  const auto Nsample = config.Nsample;
  const auto Hnode_plus1 = config.Hnode + 1;

  // Step 1: Generate Gaussian noise
  Eigen::Tensor<double, 3> eps_Y = GenerateGaussianNoise(Nsample, Hnode_plus1, nu);

  // Step 2: Scale and shift the noise
  Eigen::Tensor<double, 3> Y0s(Nsample, Hnode_plus1, nu);
  for (int i = 0; i < Nsample; ++i) {
    for (int j = 0; j < Hnode_plus1; ++j) {
      for (int k = 0; k < nu; ++k) {
        Y0s(i, j, k) = eps_Y(i, j, k) * noise_scale + Ybar_i(j, k);
      }
    }
  }

  // Step 3: Enforce the first control input constraint
  for (int i = 0; i < Nsample; ++i) {
    for (int k = 0; k < nu; ++k) {
      Y0s(i, 0, k) = Ybar_i(0, k);
    }
  }

  // Step 4: Append Ybar_i to Y0s
  Eigen::Tensor<double, 3> Y0s_appended(Nsample + 1, Hnode_plus1, nu);
  for (int i = 0; i < Nsample; ++i) {
    for (int j = 0; j < Hnode_plus1; ++j) {
      for (int k = 0; k < nu; ++k) {
        Y0s_appended(i, j, k) = Y0s(i, j, k);
      }
    }
  }
  for (int j = 0; j < Hnode_plus1; ++j) {
    for (int k = 0; k < nu; ++k) {
      Y0s_appended(Nsample, j, k) = Ybar_i(j, k);
    }
  }

  // Step 5: Clip values to the range [-1.0, 1.0]
  for (int i = 0; i < Nsample + 1; ++i) {
    for (int j = 0; j < Hnode_plus1; ++j) {
      const auto& bounds = planner_->policy.action_limits;
      for (int k = 0; k < nu; ++k) {
        Y0s_appended(i, j, k) = std::max(bounds[2 * k], std::min(bounds[2 * k + 1], Y0s_appended(i, j, k)));
        //Y0s_appended(i, j, k) = std::max(-1.0, std::min(1.0, Y0s_appended(i, j, k)));
      }
    }
  }

  // Step 6: Flatten the 3D tensor into a 2D Eigen::MatrixXd
  std::vector<Eigen::MatrixXd> Y0s_flat;
  const int N = Nsample + 1; // Number of matrices
  const int T = Hnode_plus1; // Rows in each matrix
  const int D = nu; // Columns in each matrix

  Y0s_flat.reserve(N);
  for (int n = 0; n < N; ++n) {
    Eigen::MatrixXd mat(T, D);
    for (int t = 0; t < T; ++t) {
      for (int d = 0; d < D; ++d) {
        mat(t, d) = Y0s_appended(n, t, d);
      }
    }
    Y0s_flat.emplace_back(std::move(mat));
  }

  // STEP 7: Rollout then call Estimate_mu_0tm1
  // ROLOUT to get rewards
  // update [nominal_policy] <- [policy] from the previous run
  planner_->Rollouts(N, Y0s_flat, horizon, pool);

  // STEP 8: Trajectories' [residual] -> weights
  Eigen::MatrixXd rewards = Eigen::MatrixXd::Zero(N, T);
  for (int i = 0; i < N; ++i) {
    const auto& res = planner_->trajectory[i]->residual;
    std::vector<double> rew(res.size(), 0);
    std::transform(res.begin(), res.end(), rew.begin(), [](double x) { return std::exp(-x); });
    rewards.row(i) = Eigen::Map<Eigen::RowVectorXd>(rew.data(), T);
  }
  Eigen::VectorXd weights = Estimate_mu_0tm1(rewards);

  // STEP 9: [weights] -> Ybar
  Eigen::MatrixXd weight_mat = SoftmaxUpdate(weights, Y0s_flat);
  Eigen::MatrixXd Ybar = WeightedAverage(weight_mat, Y0s);
  //auto qbar = WeightedAverage(weights, qss);
  //auto qdbar = WeightedAverage(weights, qdss);
  //auto xbar = WeightedAverage(weights, xss); // planner_->trajectory[i]->state
  return Ybar;
}
} // namespace mjpc