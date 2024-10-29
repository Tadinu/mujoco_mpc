#include "mjpc/optimizers/stochastic_optimizer.h"

#include <mujoco/mujoco.h>

#include <iostream>

using namespace std;
using namespace stograd;

int main(int argc, char* argv[]) {
  const size_t N = 5;
  const size_t D = 2;

  // ground truth beta
  vector<double> beta(D, 0);

  cout << "beta: [" << beta[0] << ", " << beta[1] << "]" << endl << endl;

  cout << "Populate example data ..." << endl << endl;

  vector<vector<double>> X(N);
  vector<double> y(N);

  X[0].resize(D);
  X[0][0] = -2.0;
  X[0][1] = -1.0;

  X[1].resize(D);
  X[1][0] = 2.0;
  X[1][1] = 0.0;

  X[2].resize(D);
  X[2][0] = 1.0;
  X[2][1] = 3.0;

  X[3].resize(D);
  X[3][0] = 0.0;
  X[3][1] = -1.0;

  X[4].resize(D);
  X[4][0] = 1.0;
  X[4][1] = 2.0;

  // y = X \beta
  for (size_t i = 0; i < N; ++i) {
    y[i] = stograd::dot_product(X[i], beta);
  }

  // Estimate beta by using stochastic gradient descent
  // to minimize the squared error objective function
  run_opt<stepper::Constant<>>("Stochastic gradient descent", X, y, N, D);

  run_opt<stepper::Constant<>>("Finite difference approximation", X, y, N, D);

  run_opt<stepper::Momentum<>>("Momentum", X, y, N, D);

  run_opt<stepper::RMSprop<>>("RMSprop", X, y, N, D);

  run_opt<stepper::AdaDelta<>>("AdaDelta", X, y, N, D);

  run_opt<stepper::Adam<>>("Adam", X, y, N, D);

  run_opt<stepper::AdaMax<>>("AdaMax", X, y, N, D);

  run_opt<stepper::YamAdam<>>("YamAdam", X, y, N, D);

  run_opt<stepper::AMSGrad<>>("AMSGrad", X, y, N, D);

  run_opt<stepper::Yogi<>>("Yogi", X, y, N, D);

  return 0;
}
