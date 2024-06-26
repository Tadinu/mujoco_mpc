/**
 * This file is part of RMPCPP
 *
 * Copyright (C) 2020 Michael Pantic <mpantic at ethz dot ch>
 *
 * RMPCPP is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * RMPCPP is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with RMPCPP. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef RMPCPP_CORE_POLICY_VALUE_H_
#define RMPCPP_CORE_POLICY_VALUE_H_

#include <Eigen/Dense>

// https://eigen.tuxfamily.org/dox/group__DenseDecompositionBenchmark.html
#define RMP_USE_QR_INVERSE_MATRIX (0)
/* https://www.naukri.com/code360/library/understanding-svd-decomposition
 * JacobiSVD: For small matrices, two-sided Jacobi iterations are quickly implemented, but for bigger matrices, they take a very long time.
 * BDCSVD: Applying an upper-bidiagonalization that is still quick for large problems on top of a recursive divide-and-conquer approach.
 * -> Divide-and-conquer diagonalizes the input matrix after first reducing it to bi-diagonal form using class UpperBidiagonalization.
 */
#define RMP_USE_JACOBI_SVD_INVERSE_MATRIX (1)
#define RMP_USE_BDC_SVD_INVERSE_MATRIX (!RMP_USE_QR_INVERSE_MATRIX && !RMP_USE_JACOBI_SVD_INVERSE_MATRIX)

namespace rmpcpp {
/**
 * Evaluated policy consisting of concrete f and A
 * @tparam d Dimensionality
 */
template <int d>
class PolicyValue {
 public:
  using Matrix = Eigen::Matrix<double, d, d>;
  using Vector = Eigen::Matrix<double, d, 1>;

  PolicyValue(const Vector &f, const Matrix &A) : A_(A), f_(f) {}

  /**
   * Implemetation of addition operation.
   * Defined in Eq. 8 in [1]
   */
  PolicyValue operator+(PolicyValue &other) {
    Matrix A_combined = this->A_ + other.A_;
    Vector f_combined = pinv(A_combined) * (this->A_ * this->f_ + other.A_ * other.f_);

    return PolicyValue(f_combined, A_combined);
  }

  /**
   * Sum operator as defined in eq. 9 in [1].
   */
  static PolicyValue sum(const std::vector<PolicyValue>& RMPBases) {
    Matrix sum_ai = Matrix::Zero();
    Vector sum_ai_fi = Vector::Zero();

    // sum up terms
    for (const auto &rmpBase : RMPBases) {
      sum_ai += rmpBase.A_;
      sum_ai_fi += rmpBase.A_ * rmpBase.f_;
    }

    auto f_summed = pinv(sum_ai) * sum_ai_fi;
    return PolicyValue(f_summed, sum_ai);
  }

#if RMP_USE_QR_INVERSE_MATRIX
  /// Convenience method for pseudo-inverse
  template<int i = d, int j = d, typename TMatrix = Eigen::Matrix<double, i, j>>
  static inline TMatrix pinv(const Eigen::Matrix<double, i, j> &M) {
    return (M.completeOrthogonalDecomposition().pseudoInverse());
  }
#elif RMP_USE_JACOBI_SVD_INVERSE_MATRIX
  // https://eigen.tuxfamily.org/dox/group__LeastSquares.html
  // https://gist.github.com/javidcf/25066cf85e71105d57b6
  template<int i = d, int j = d, typename TMatrix = Eigen::Matrix<double, i, j>>
  static inline TMatrix pinv(const TMatrix &M,
                             double epsilon = std::numeric_limits<double>::epsilon()) {
#if 1
    Eigen::JacobiSVD<TMatrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // For a non-square matrix
    // Eigen::JacobiSVD<TMatrix> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = epsilon * std::max(M.cols(), M.rows()) * svd.singularValues().array().abs()(0);
    return svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
#else
    auto svd = M.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    const auto &singularValues = svd.singularValues();
    TMatrix singularValuesInv(M.cols(), M.rows());
    singularValuesInv.setZero();
    double tolerance = epsilon * std::max(M.cols(), M.rows()) * singularValues.array().abs()(0);
    for (unsigned int k = 0; k < singularValues.size(); ++k) {
      if (singularValues(k) > tolerance)
      {
        singularValuesInv(k, k) = 1.0 / singularValues(k);
      }
      else
      {
        singularValuesInv(k, k) = 0.0;
      }
    }
    return svd.matrixV() * singularValuesInv * svd.matrixU().adjoint();
#endif
  }
#elif RMP_USE_BDC_SVD_INVERSE_MATRIX
  // https://gist.github.com/pshriwise/67c2ae78e5db3831da38390a8b2a209f
  template<int i = d, int j = d, typename TMatrix = Eigen::Matrix<double, i, j>>
  static inline TMatrix pinv(const TMatrix &M,
                             double epsilon = std::numeric_limits<double>::epsilon())
  {
    Eigen::BDCSVD<TMatrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    svd.setThreshold(epsilon*std::max(M.cols(), M.rows()));
    Eigen::Index rank = svd.rank();
    TMatrix tmp = svd.matrixU().leftCols(rank).adjoint();
    tmp = svd.singularValues().head(rank).asDiagonal().inverse() * tmp;
    return svd.matrixV().leftCols(rank) * tmp;
  }
#endif

  const Matrix A_;
  const Vector f_;
};

}  // namespace rmpcpp

#endif  // RMPCPP_CORE_POLICY_VALUE_H_