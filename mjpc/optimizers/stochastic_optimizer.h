// Modified from https://github.com/djhshih/stograd
#pragma once

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

// MuJoCo
#include <mujoco/mujoco.h>

#include "mjpc/optimizers/base_optimizer.h"

namespace stograd {

using namespace std;

#ifdef DEBUG

class Optimizable {
public:
  /// Number of observations.
  virtual std::size_t nobs() const = 0;

  /// Number of parameters.
  virtual std::size_t nparams() const = 0;

  /// Accumulate gradient.
  ///
  /// Compute the gradient of the objective function of the next data point
  /// and add it to the provided current gradient vector
  ///
  /// @param grad  current gradient vector (of size `nparams`)
  virtual void accumulate(std::vector<double>& grad) = 0;

  /// Update parameter vector.
  ///
  /// To minimize the objective function, use the `substract_from` utility
  /// function to update the internal parameter vectorc with the provided
  /// delta vector. Otherwise, use the `add_to` function.
  ///
  /// @param delta  vector for updating the parameter vector
  virtual void update(const std::vector<double>& delta) = 0;

protected:
  // Disallow polymorphic deletion through a base pointer
  virtual ~Optimizable() {}
};

#endif  // DEBUG

/**
 * Substract vector xs from vector ys
 * ys -= xs
 *
 * Vectors must have the same size.

 */
template <typename T>
void subtract_from(const vector<T>& xs, vector<T>& ys) {
#if 1
  mju_subFrom(ys.data(), xs.data(), xs.size());
#else
  typename vector<T>::const_iterator xit, xend = xs.end();
  typename vector<T>::iterator yit;
  for (xit = xs.begin(), yit = ys.begin(); xit != xend; ++xit, ++yit) {
    (*yit) -= (*xit);
  }
#endif
}

/**
 * Add vector xs to vector ys
 * ys += xs
 *
 * Vectors must have the same size.
 */
template <typename T>
void add_to(const vector<T>& xs, vector<T>& ys) {
#if 1
  mju_addTo(ys.data(), xs.data(), xs.size());
#else
  typename vector<T>::const_iterator xit, xend = xs.end();
  typename vector<T>::iterator yit;
  for (xit = xs.begin(), yit = ys.begin(); xit != xend; ++xit, ++yit) {
    (*yit) += (*xit);
  }
#endif
}

/**
 * Dot product of xs and ys.
 *
 * Vectors must have the same size.
 */
template <typename T>
T dot_product(const vector<T>& xs, const vector<T>& ys) {
#if 1
  return mju_dot(xs.data(), ys.data(), ys.size());
#else
  typename vector<T>::const_iterator xit, yit, xend = xs.end();
  T p = 0;
  for (xit = xs.begin(), yit = ys.begin(); xit != xend; ++xit, ++yit) {
    p += (*xit) * (*yit);
  }
  return p;
#endif
}

/**
 * Root of sum of squares.
 */
template <typename T>
double rss(const vector<T>& xs) {
#if 1
  return mju_sqrt(mju_sum(xs.data(), xs.size()));
#else
  typename vector<T>::const_iterator xit, xend = xs.end();
  double r = 0.0;
  for (xit = xs.begin(); xit != xend; ++xit) {
    double x = *xit;
    r += x * x;
  }
  return sqrt(r);
#endif
}

/**
 * Sign of nuemric value.
 */
template <typename T>
int sign(T t) {
#if 1
  return mju_sign(t);
#else
  return (T(0) < t) - (t < T(0));
#endif
}

/**
 * Sigmoid function.
 */
template <typename T>
T logistic(T t) {
  return 1 / (1 + exp(-t));
}

namespace stepper {

/// Non-adaptive
template <typename TReal = double, typename = std::enable_if_t<std::is_floating_point_v<TReal>>>
struct Constant {
  // learning rate
  TReal r;

  Constant(TReal rate = 0.01) : r(rate) {}

  TReal operator()(TReal g) { return r * g; }
};

/// Momemtum
template <typename TReal = double, typename = std::enable_if_t<std::is_floating_point_v<TReal>>>
struct Momentum {
  // base learning rate
  TReal r;

  // hyperparameter
  TReal b;

  // first moment
  TReal v;

  Momentum(TReal rate = 0.001, TReal beta = 0.9) : r(rate), b(beta), v(0.0) {}

  TReal operator()(TReal g) {
    // update moment
    v = b * v + g;

    return r * v;
  }
};

/// RMSprop
/// equivalent to ADAM with beta2=0 and no bias correction
template <typename TReal = double, typename = std::enable_if_t<std::is_floating_point_v<TReal>>>
struct RMSprop {
  // base learning rate
  TReal r;

  // hyperparameters
  TReal a, e;

  // second moment
  TReal v;

  RMSprop(TReal rate = 0.001, TReal alpha = 0.9, TReal epsilon = 1e-6)
      : r(rate), a(alpha), e(epsilon), v(0.0) {}

  TReal operator()(TReal g /*grad*/) {
    // update moment
    v = a * v + (1 - a) * g * g;

    // NB epsilon is intentionally placed outside sqrt
    // https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    return r * g / (sqrt(v) + e);
  }
};

/// AdaDelta
template <typename TReal = double, typename = std::enable_if_t<std::is_floating_point_v<TReal>>>
struct AdaDelta {
  // hyperparameters
  TReal b, e;

  // second moments of gradient and delta
  TReal v, s;

  // previous delta
  TReal d;

  AdaDelta(TReal beta = 0.95, TReal epsilon = 1e-6) : b(beta), e(epsilon), v(0.0), s(0.0), d(0.0) {}

  TReal operator()(TReal g) {
    // update moments
    v = b * v + (1 - b) * g * g;
    s = b * s + (1 - b) * d * d;

    d = sqrt(s + e) / sqrt(v + e) * g;

    return d;
  }
};

/// ADAM
template <typename TReal = double, typename = std::enable_if_t<std::is_floating_point_v<TReal>>>
struct Adam {
  // base learning rate
  TReal r;

  // hyperparameters
  TReal b1, b2, e;

  // bias correct the moments
  // (this can cause v to blow up too quick, causing premature stopping)
  bool debias;

  // first and second moments
  TReal m, v;

  // timestep
  unsigned long int t;

  Adam(TReal rate = 0.001, TReal beta1 = 0.9, TReal beta2 = 0.999, TReal epsilon = 1e-3)
      : r(rate), b1(beta1), b2(beta2), e(epsilon), debias(false), m(0.0), v(0.0), t(0) {}

  TReal operator()(TReal g) {
    ++t;

    // update moments
    m -= (1 - b1) * (m - g);
    v -= (1 - b2) * (v - g * g);
    // equivalently,
    // m = b1*m + (1 - b1)*g;
    // v = b2*v + (1 - b2)*g*g;

    if (debias) {
      // correct for bias in moments
      m = m / (1 - pow(b1, t));
      v = v / (1 - pow(b2, t));
    }

    return r * m / (sqrt(v) + e);
  }
};

/// AdaMax
template <typename TReal = double, typename = std::enable_if_t<std::is_floating_point_v<TReal>>>
struct AdaMax {
  // base learning rate
  TReal r;

  // hyperparameters
  TReal b1, b2;

  // first and second moments
  TReal m, v;

  AdaMax(TReal rate = 0.001, TReal beta1 = 0.9, TReal beta2 = 0.999)
      : r(rate), b1(beta1), b2(beta2), m(0.0), v(0.0) {}

  TReal operator()(TReal g) {
    // update moments
    m -= (1 - b1) * (m - g);
    v = max(b2 * v, abs(g));

    return r * m / v;
  }
};

/// YamAdam
template <typename TReal = double, typename = std::enable_if_t<std::is_floating_point_v<TReal>>>
struct YamAdam {
  // hyperparameter
  TReal e;

  // first moment of gradient
  TReal m;

  // second centered moment of gradient
  TReal v;

  // second moment of delta
  TReal s;

  // previous and current delta
  TReal dp, d;

  // exponential moving average coefficient
  TReal b;

  YamAdam(TReal epsilon = 1e-6) : e(epsilon), m(0.0), v(0.0), s(0.0), d(0.0), b(0.0) {}

  TReal operator()(TReal g) {
    dp = d;

    // update moments
    m = b * m + (1 - b) * g;
    v = b * v + (1 - b) * (g - m) * (g - m);
    s = b * s + (1 - b) * d * d;

    d = sqrt(s + e) / sqrt(v + e) * m;

    // update coefficient
    // NB difference from Kazunori Yamada's reference implementation:
    //    the coefficient upate uses the L1 norm of vector d,
    //    and there is only one coefficient shared across parameter
    //    dimensions
    b = logistic((abs(d) + e) / (abs(dp) + e)) - e;

    return d;
  }
};

/// AMSGrad
template <typename TReal = double, typename = std::enable_if_t<std::is_floating_point_v<TReal>>>
struct AMSGrad {
  // base learning rate
  TReal r;

  // hyperparameters
  TReal b1, b2, e;

  // first and second moments
  TReal m, v;

  AMSGrad(TReal rate = 0.001, TReal beta1 = 0.9, TReal beta2 = 0.999, TReal epsilon = 1e-3)
      : r(rate), b1(beta1), b2(beta2), e(epsilon), m(0.0), v(0.0) {}

  TReal operator()(TReal g) {
    // update first moment
    m = b1 * m + (1 - b1) * g;
    // only update second moment if it becomes larger
    v = max(v, b2 * v + (1 - b2) * g * g);

    return r * m / (sqrt(v) + e);
  }
};

/// YOGI
template <typename TReal = double, typename = std::enable_if_t<std::is_floating_point_v<TReal>>>
struct Yogi {
  // base learning rate
  TReal r;

  // hyperparameters
  TReal b1, b2, e;

  // bias correct the moments
  // (this can cause v to blow up too quick, causing premature stopping)
  bool debias;

  // first and second moments
  TReal m, v;

  // timestep
  unsigned long int t;

  Yogi(TReal rate = 0.01, TReal beta1 = 0.9, TReal beta2 = 0.999, TReal epsilon = 1e-3)
      : r(rate), b1(beta1), b2(beta2), e(epsilon), debias(false), m(0.0), v(0.0), t(0) {}

  TReal operator()(TReal g) {
    ++t;

    // update moments
    m -= (1 - b1) * (m - g);
    v -= (1 - b2) * sign(v - g * g) * g * g;

    if (debias) {
      // correct for bias in moments
      m = m / (1 - pow(b1, t));
      v = v / (1 - pow(b2, t));
    }

    return r * m / (sqrt(v) + e);
  }
};

}  // namespace stepper

/**
 * Optimize an objective function.
 *
 * Minimize (maximize) an objective function by stochastic gradient descent
 * (ascent) to a possibly local (but hopefully global) optimum.
 *
 * The TOptimizable type is implicitly required to implement the TOptimizable
 * interface class given above. For production code, we use template instead
 * of an abstract class for superior runtime speed.
 *
 * @param op      an object of a class that implements the implicit
 *                TOptimizable interface
 * @param stepf   functor object for adapting the gradient into a step
 *                (e.g. stepper::Constant, stepper::Adam)
 * @param bsize   batch size
 * @param nepochs number of passes through the data
 * @param eps     threshold on gradient norm for early convergence
 * @return  number of passes through the data,
 *          negated if gradient did not converge to zero early
 */
template <typename TOptimizable, typename F, typename TReal = double>
int optimize(TOptimizable& op, const F& stepf, size_t bsize, size_t nepochs, TReal eps = 1e-4) {
  size_t nobs = op.nobs();
  size_t nparams = op.nparams();

  // number of batches
  size_t nbatches = nobs / bsize;

  // number of leftover data points
  size_t nremnant = nobs - (nbatches * bsize);

  // early convergence of gradient to zero
  bool converged = false;

  // initialize a stepper for each parameter (to avoid reusing possibly cached values in each [stepf])
  vector<F> stepfs(nparams, stepf);

  cout << "nepochs: " << nepochs << endl;
  cout << "nbatches: " << nbatches << endl;
  cout << "beta:";
  MJPC_PRINT(op.beta_);

  size_t a;
  for (a = 0; a < nepochs; ++a) {
    // overall gradient across all batches
    vector<TReal> odelta(nparams, 0.0);

    for (size_t b = 0; b < nbatches; ++b) {
      // last batch will include any leftover data points
      size_t _bsize;
      if (b == nbatches - 1) {
        _bsize = bsize + nremnant;
      } else {
        _bsize = bsize;
      }

      vector<TReal> grad(nparams, 0.0);

      // iterate through data points in a batch to accumulate the gradient
      for (size_t i = 0; i < _bsize; ++i) {
        op.accumulate(grad);
      }

      // call the steppers with the normalized gradient to compute the steps
      // reusing grad vector for storing the delta values
      typename vector<TReal>::iterator it;
      typename vector<TReal>::const_iterator end = grad.end();
      typename vector<F>::iterator sit;
      for (it = grad.begin(), sit = stepfs.begin(); it != end; ++it, ++sit) {
        // normalize the gradient by the batch size
        double g = (*it) / _bsize;
        (*it) = (*sit)(g);
      }
      vector<TReal>& delta = grad;

      // update the parameter vector
      op.update(delta);

      // accumulate the overall delta vector
      add_to(delta, odelta);
    }  // nbatches

    if (rss(odelta) < eps) {
      converged = true;
      break;
    }
  }  // nepochs

  const int out_epochs = converged ? a : -a;
  cout << "elasped epochs: " << out_epochs << endl;
  cout << "beta_hat:";
  MJPC_PRINT(op.beta_);
  cout << "model vals: " << op.m_.y_ << std::endl;
  return out_epochs;
}

/**
 * Approximate gradient by central finite difference.
 *
 *
 * @param fn  functor object that accepts vector<TReal>& and returns TReal;
 *            evaluation fn(x) must have no side-effect
 * @param x   point at which to evaluate the gradient
 * @param g   uninitialized out parameter that will hold the evaluated value
 * @param step  step size
 */
template <typename F, typename TReal = double>
void finite_difference_gradient(const F& fn, const vector<TReal>& x, vector<TReal>& g, TReal step = 1e-6) {
  size_t D = x.size();
  g.reserve(D);
  for (size_t d = 0; d < D; ++d) {
    vector<TReal> xp(x);
    xp[d] += step;

    vector<TReal> xm(x);
    xm[d] -= step;

    g.push_back((fn(xp) - fn(xm)) / (2 * step));
  }
}

struct Model {
  /// Rows of predictor variable values
  vector<vector<double>> X_;

  /// Observed response variable values
  vector<double> y_;

  /// Index of the current observation
  size_t i = 0;

  Model() = default;
  Model(const vector<vector<double>>& X, const vector<double>& y) : X_(X), y_(y), i(0) {}

  /// Return squared error
  /// (y - X \beta)^\top (y - X \beta)
  /// This is not required if exact analytic gradient is available
  double objective(const vector<double>& beta) const {
    double e = y_[i] - dot_product(X_[i], beta);
    return e * e;
  }

  /// Return gradient w.r.t. beta in vector g
  /// g is a out parameter that must be passed in uninitialized
  /// ( -2 (y - X \beta)^\top X )^\top
  virtual void gradient(const vector<double>& beta, vector<double>& g) const {
    const vector<double>& xi = X_[i];
    size_t D = xi.size();
    g.reserve(D);

    double e = (y_[i] - dot_product(xi, beta));
    for (size_t d = 0; d < D; ++d) {
      g.push_back(-2.0 * e * xi[d]);
    }
  }

  /// Move onto next observation
  void next() {
    ++i;
    if (i == y_.size()) {
      i = 0;
    }
  }

  double operator()(const vector<double>& beta) const { return objective(beta); }
};

struct Optimizable {
  // Observations no
  size_t N_;

  // Params no
  size_t D_;

  Model m_;
  vector<double> beta_;

  Optimizable() = default;
  Optimizable(size_t N, size_t D, const Model& m) : N_(N), D_(D), m_(m), beta_(D, 0) {}

  size_t nobs() const { return N_; }

  size_t nparams() const { return D_; }

  /// Compute gradient based on observation i and
  /// accumulate the current gradient
  void accumulate(vector<double>& grad) {
    vector<double> gradi;
    m_.gradient(beta_, gradi);

    add_to(gradi, grad);

    m_.next();
  }

  /// Update beta based on provided delta vector
  void update(const vector<double>& delta) { subtract_from(delta, beta_); }
};

namespace finite_difference {

// Alternative Model using finite difference approximation for the gradient
struct Model : public stograd::Model {
  Model(const vector<vector<double>>& X, const vector<double>& y) : stograd::Model(X, y) {}

  void gradient(const vector<double>& beta, vector<double>& g) const {
    finite_difference_gradient(*this, beta, g);
  }
};

}  // namespace finite_difference

class RMSpropOptimizer : public BaseOptimizer {
public:
  RMSpropOptimizer(mjModel* mj_model, mjData* mj_data, mjpc::Task* mj_task, mjpc::Planner* mj_planner)
      : BaseOptimizer(mj_model, mj_data, mj_task, mj_planner) {
    m = Model({}, mj_planner_->GetNominalPolicyValues());
  }
  RMSpropOptimizer(const vector<vector<double>>& X, const vector<double>& y, size_t N, size_t D, double step)
      : m(X, y), opt(N, D, m), step(step) {}
  void optimize() override {
    cout << "RMSprop ..." << endl;
    stograd::optimize(opt, step, 2 /* batch size */, 1000 /*nepochs*/, 1e-3 /*eps*/);
  }

  std::vector<double> opt_vals() const override { return opt.beta_; }

protected:
  Model m;
  Optimizable opt;
  stepper::RMSprop<double> step;
};
using RMSpropOptimizerPtr = std::shared_ptr<RMSpropOptimizer>;

template <typename T, typename TReal = double>
static void run_opt(const char* text, const vector<vector<TReal>>& X, const vector<TReal>& y, const size_t N,
                    const size_t D) {
  cout << "Running with " << text << " ..." << endl;
  Model m(X, y);
  Optimizable opt(N, D, m);
  T st(0.01);
  optimize(opt, st, 2, 1000, 1e-3);
}
}  // namespace stograd
