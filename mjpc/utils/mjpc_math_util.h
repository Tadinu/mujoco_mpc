#pragma once

#include <random>

// mjpc
#include "mjpc/utils/mjpc_core_util.h"

namespace mjpc {
struct Random {
  static int seed;
  // PRN, random seed for the random number engine
  static std::random_device rd;

  // Standard mersenne_twister_engine seeded with rd()
  static std::mt19937_64 gen;

  template <typename T>
  static T rand(const T min, const T max) {
    if constexpr (std::is_integral_v<T>) {
      return std::uniform_int_distribution<T>(min, max)(gen);
    } else if constexpr (std::is_floating_point_v<T>) {
      return std::uniform_real_distribution<T>(min, max)(gen);
    } else {
      return {};
    }
  }

  static double rand() { return rand<double>(0.f, 1.f); }
  static bool rand_bool() { return rand() < 0.5f; }
};

// HELPER FUNCTIONS ---------------------
static constexpr mjtNum POSITION_ZERO[3] = {0, 0, 0};
static constexpr mjtNum QUAT_IDENTITY[4] = {1, 0, 0, 0};
static constexpr mjtNum POSE_IDENTITY[7] = {0, 0, 0, 1, 0, 0, 0};
// NOTE: Ones prefixed with "Mj" are either copied (while waiting to be released) or wrapper of/modified from mju_ API
static std::vector<mjtNum> MjuIdentityMatrix(int n) {
  std::vector<mjtNum> identity(n * n, 0);
  mju_eye(identity.data(), n);
  return identity;
}

// Ref: mju_quatZ2Vec() calculates quaternion from Z-vector to a vector
static void MjuQuatFromVectors(mjtNum quat[4], const mjtNum vec1[3], const mjtNum vec2[3]) {
  mjtNum axis[3], a, vec2n[3] = {vec2[0], vec2[1], vec2[2]};

  // set default result to no-rotation quaternion
  quat[0] = 1;
  mju_zero3(quat + 1);

  // normalize vector; if too small, no rotation
  if (mju_normalize3(vec2n) < mjMINVAL) {
    return;
  }

  // compute angle and axis
  mju_cross(axis, vec1, vec2);
  a = mju_normalize3(axis);

  // almost parallel
  if (mju_abs(a) < mjMINVAL) {
    // opposite: 180 deg rotation around x axis
    if (mju_dot3(vec2, vec1) < 0) {
      quat[0] = 0;
      quat[1] = 1;
    }

    return;
  }

  // make quaternion from angle and axis
  a = mju_atan2(a, mju_dot3(vec2, vec1));
  mju_axisAngle2Quat(quat, axis, a);
}

static void MjuNormalToQuat(mjtNum quat[4], const mjtNum norm[3]) {
  // Reference direction (world z-axis)
  const mjtNum z_ref[3] = {0.0, 0.0, 1.0};

  // Compute rotation axis (cross product)
  mjtNum axis[3] = {
      z_ref[1] * norm[2] - z_ref[2] * norm[1],
      z_ref[2] * norm[0] - z_ref[0] * norm[2],
      z_ref[0] * norm[1] - z_ref[1] * norm[0]
  };

  // Compute rotation angle, clipped for numerical stability
  double dot = std::max(-1.0, std::min(1.0, z_ref[0] * norm[0] + z_ref[1] * norm[1] + z_ref[2] * norm[2]));
  double angle = std::acos(dot);

  // Special case: (0, 0, -1) normal
  if (std::abs(norm[0]) < 1e-6 && std::abs(norm[1]) < 1e-6 && std::abs(norm[2] + 1) < 1e-6) {
    // 180-degree around x-axis
    memcpy(quat, (double[]){0.0, 1.0, 0.0, 0.0}, 4 * sizeof(mjtNum));
  }

  // Normalize rotation axis
  if (std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]) > 1e-6) {
    // Avoid divide by zero
    mju_normalize3(axis);
  } else {
    // Default axis if normal is aligned with z_ref
    memcpy(axis, (double[]){1.0, 0.0, 0.0}, sizeof(axis));
  }

  // Convert [axis, angle] -> quat
  mju_axisAngle2Quat(quat, axis, angle);
}

// Ref: https://github.com/google-deepmind/mujoco/blob/main/src/user/user_util.h
// convert global to local axis relative to given frame
static void MjuLocalAxis(double* al, const double* ag, const double* quat) {
  double mat[9];
  double qneg[4] = {quat[0], -quat[1], -quat[2], -quat[3]};
  mju_quat2Mat(mat, qneg);
  mju_mulMatVec3(al, ag, mat);
}

// Ref: mj_local2Global()
// convert global to local position relative to given frame
static void MjuLocalPos(double* pl, const double* pg, const double* pos, const double* quat) {
  double a[3] = {pg[0] - pos[0], pg[1] - pos[1], pg[2] - pos[2]};
  MjuLocalAxis(pl, a, quat);
}

// compute quaternion rotation from parent to child
static void MjuLocalQuat(double* local, const double* child, const double* parent) {
  double pneg[4] = {parent[0], -parent[1], -parent[2], -parent[3]};
  mju_mulQuat(local, pneg, child);
}

// Ref: mj_fullM
// Convert sparse inertia matrix M into full (i.e. dense) matrix.
static void MjuFullMatrix(const mjModel* m, mjtNum* dst, const mjtNum* M /* inertial matrix: qM*/,
                          int start_idx, int size) {
  int adr = 0;
  mju_zero(dst, size * size);

  for (int i = start_idx; i < start_idx + size; ++i) {
    int _i = i - start_idx;
    int j = i;
    while (j >= 0) {
      int _j = j - start_idx;
      dst[_i * size + _j] = M[adr];
      dst[_j * size + _i] = M[adr];
      j = m->dof_parentid[j];
      adr++;
    }
  }
}

// Ref: [engine_support.h] - mj_bodyChain()
static int MjuBodyChain(const mjModel* m, int* chain, int body, int base_body = 0) {
  // simple body
  if (m->body_simple[body]) {
    int dofnum = m->body_dofnum[body];
    for (int i = 0; i < dofnum; i++) {
      chain[i] = m->body_dofadr[body] + i;
    }
    return dofnum;
  }

  // general case
  else {
    // skip fixed bodies
    while (body && !m->body_dofnum[body]) {
      body = m->body_parentid[body];
    }

    // not movable: empty chain
    if (body == base_body) {
      return 0;
    }

    // intialize last dof
    int da = m->body_dofadr[body] + m->body_dofnum[body] - 1;
    int NV = 0;

    // construct chain from child to parent
    //mjpc::print(body, mj_id2name(m, mjOBJ_BODY, body), "dof_adr", m->body_dofadr[body]);
    //mjpc::print(base_body, mj_id2name(m, mjOBJ_BODY, base_body), "dof_adr", m->body_dofadr[base_body]);
    while ((da > -1) && (da >= m->body_dofadr[base_body])) {
      //mjpc::print(da, mj_id2name(m, mjOBJ_JOINT, m->dof_jntid[da]), "body", m->dof_bodyid[da]);
      chain[NV++] = da;
      if (m->dof_bodyid[da] == base_body + 1) {
        break;
      }
      da = m->dof_parentid[da];
    }

    // reverse order of chain: make it increasing
    for (int i = 0; i < NV / 2; i++) {
      int tmp = chain[i];
      chain[i] = chain[NV - i - 1];
      chain[NV - i - 1] = tmp;
    }

    return NV;
  }
}

static int MjuBodyChainDofNum(const mjModel* m, int body, int base_body = 0) {
  // simple body
  if (m->body_simple[body]) {
    return m->body_dofnum[body];
  }

  // general case
  else {
    // skip fixed bodies
    while (body && !m->body_dofnum[body]) {
      body = m->body_parentid[body];
    }

    // not movable: empty chain
    if (body == base_body) {
      return 0;
    }

    // intialize last dof
    int da = m->body_dofadr[body] + m->body_dofnum[body] - 1;
    int NV = 0;

    // construct chain from child to parent
    //mjpc::print(body, mj_id2name(m, mjOBJ_BODY, body), "dof_adr", m->body_dofadr[body]);
    //mjpc::print(base_body, mj_id2name(m, mjOBJ_BODY, base_body), "dof_adr", m->body_dofadr[base_body]);
    while ((da > -1) && (da >= m->body_dofadr[base_body])) {
      //mjpc::print(da, mj_id2name(m, mjOBJ_JOINT, m->dof_jntid[da]), "body", m->dof_bodyid[da]);
      NV++;
      if (m->dof_bodyid[da] == base_body + 1) {
        break;
      }
      da = m->dof_parentid[da];
    }
    return NV;
  }
}

// Ref: [engine_support.h] - mj_jacSparseSimple()
// sparse Jacobian difference for simple body contacts
static void MjuJacSparseSimple(const mjModel* m, const mjData* d,
                               mjtNum* jacdifp, mjtNum* jacdifr, const mjtNum* point,
                               int body, int flg_second, int NV, int start) {
  // compute point-com offset
  mjtNum offset[3];
  mju_sub3(offset, point, d->subtree_com + 3 * m->body_rootid[body]);

  // skip fixed body
  if (!m->body_dofnum[body]) {
    return;
  }

  // process dofs
  int ci = start;
  int end = m->body_dofadr[body] + m->body_dofnum[body];
  for (int da = m->body_dofadr[body]; da < end; da++) {
    mjtNum* cdof = d->cdof + 6 * da;

    // construct rotation jacobian
    if (jacdifr) {
      // plus sign
      if (flg_second) {
        jacdifr[ci + 0 * NV] = cdof[0];
        jacdifr[ci + 1 * NV] = cdof[1];
        jacdifr[ci + 2 * NV] = cdof[2];
      }

      // minus sign
      else {
        jacdifr[ci + 0 * NV] = -cdof[0];
        jacdifr[ci + 1 * NV] = -cdof[1];
        jacdifr[ci + 2 * NV] = -cdof[2];
      }
    }

    // construct translation jacobian (correct for rotation)
    if (jacdifp) {
      mjtNum tmp[3];
      mju_cross(tmp, cdof, offset);

      // plus sign
      if (flg_second) {
        jacdifp[ci + 0 * NV] = (cdof[3] + tmp[0]);
        jacdifp[ci + 1 * NV] = (cdof[4] + tmp[1]);
        jacdifp[ci + 2 * NV] = (cdof[5] + tmp[2]);
      }

      // plus sign
      else {
        jacdifp[ci + 0 * NV] = -(cdof[3] + tmp[0]);
        jacdifp[ci + 1 * NV] = -(cdof[4] + tmp[1]);
        jacdifp[ci + 2 * NV] = -(cdof[5] + tmp[2]);
      }
    }

    // advance jacdif counter
    ci++;
  }
}

// Ref: [engine_support.h] - mj_jacSparse()
static void MjuJacSparse(const mjModel* m, const mjData* d,
                         mjtNum* jacp, mjtNum* jacr, const mjtNum* point, int body,
                         int NV, const int* chain, int base_body_id = 0) {
  int da, ci;
  mjtNum offset[3], tmp[3], *cdof = d->cdof;

  // clear jacobians
  if (jacp) {
    mju_zero(jacp, 3 * NV);
  }
  if (jacr) {
    mju_zero(jacr, 3 * NV);
  }

  // compute point-com offset
  mju_sub3(offset, point, d->subtree_com + 3 * m->body_rootid[body]);

  // skip fixed bodies
  while (body && !m->body_dofnum[body]) {
    body = m->body_parentid[body];
  }

  // no movable body found: nothing to do
  if (!body) {
    return;
  }

  // get last dof that affects this (as well as the original) body
  da = m->body_dofadr[body] + m->body_dofnum[body] - 1;

  // start and the end of the chain (chain is in increasing order)
  ci = NV - 1;

  // backward pass over dof ancestor chain
  while ((da > -1) && (da >= m->body_dofadr[base_body_id])) {
    // find chain index for this dof
    while (ci >= 0 && chain[ci] > da) {
      ci--;
    }

    if (m->dof_bodyid[da] == base_body_id + 1) {
      break;
    }

    // make sure we found it; SHOULD NOT OCCUR
    if (chain[ci] != da) {
      print("dof index %d not found in chain", da);
    }

    // construct rotation jacobian
    if (jacr) {
      jacr[ci] = cdof[6 * da];
      jacr[ci + NV] = cdof[6 * da + 1];
      jacr[ci + 2 * NV] = cdof[6 * da + 2];
    }

    // construct translation jacobian (correct for rotation)
    if (jacp) {
      mju_cross(tmp, cdof + 6 * da, offset);

      jacp[ci] = cdof[6 * da + 3] + tmp[0];
      jacp[ci + NV] = cdof[6 * da + 4] + tmp[1];
      jacp[ci + 2 * NV] = cdof[6 * da + 5] + tmp[2];
    }

    // advance to parent dof
    da = m->dof_parentid[da];
  }
}

// Ref: [engine_support.h] - mj_mergeChain()
// merge dof chains for two bodies
static int MjuMergeChain(const mjModel* m, int* chain, int b1, int b2) {
  int da1, da2, NV = 0;

  // skip fixed bodies
  while (b1 && !m->body_dofnum[b1]) {
    b1 = m->body_parentid[b1];
  }
  while (b2 && !m->body_dofnum[b2]) {
    b2 = m->body_parentid[b2];
  }

  // neither body is movable: empty chain
  if (b1 == 0 && b2 == 0) {
    return 0;
  }

  // initialize last dof address for each body
  da1 = m->body_dofadr[b1] + m->body_dofnum[b1] - 1;
  da2 = m->body_dofadr[b2] + m->body_dofnum[b2] - 1;

  // merge chains
  while (da1 >= 0 || da2 >= 0) {
    chain[NV] = mjMAX(da1, da2);
    if (da1 == chain[NV]) {
      da1 = m->dof_parentid[da1];
    }
    if (da2 == chain[NV]) {
      da2 = m->dof_parentid[da2];
    }
    NV++;
  }

  // reverse order of chain: make it increasing
  for (int i = 0; i < NV / 2; i++) {
    int tmp = chain[i];
    chain[i] = chain[NV - i - 1];
    chain[NV - i - 1] = tmp;
  }

  return NV;
}


// Ref: [engine_support.h] - mj_mergeChainSimple()
// merge dof chains for two simple bodies
static int MjuMergeChainSimple(const mjModel* m, int* chain, int b1, int b2) {
  // swap bodies if wrong order
  if (b1 > b2) {
    int tmp = b1;
    b1 = b2;
    b2 = tmp;
  }

  // init
  int n1 = m->body_dofnum[b1], n2 = m->body_dofnum[b2];

  // both fixed: nothing to do
  if (n1 == 0 && n2 == 0) {
    return 0;
  }

  // copy b1 dofs
  for (int i = 0; i < n1; i++) {
    chain[i] = m->body_dofadr[b1] + i;
  }

  // copy b2 dofs
  for (int i = 0; i < n2; i++) {
    chain[n1 + i] = m->body_dofadr[b2] + i;
  }

  return (n1 + n2);
}


// Ref: [engine_support.h] - mj_jacDifPair()
// dense or sparse Jacobian difference for two body points: pos2 - pos1, global
static int MjuJacDifPair(const mjModel* m, const mjData* d, int* chain,
                         int b1, int b2, const mjtNum pos1[3], const mjtNum pos2[3],
                         mjtNum* jac1p, mjtNum* jac2p, mjtNum* jacdifp,
                         mjtNum* jac1r, mjtNum* jac2r, mjtNum* jacdifr) {
  int issimple = (m->body_simple[b1] && m->body_simple[b2]);
  int issparse = mj_isSparse(m);
  int NV = m->nv;

  // skip if no DOFs
  if (!NV) {
    return 0;
  }

  // construct merged chain of body dofs
  if (issparse) {
    if (issimple) {
      NV = MjuMergeChainSimple(m, chain, b1, b2);
    } else {
      NV = MjuMergeChain(m, chain, b1, b2);
    }
  }

  // skip if empty chain
  if (!NV) {
    return 0;
  }

  // sparse case
  if (issparse) {
    // simple: fast processing
    if (issimple) {
      // first body
      MjuJacSparseSimple(m, d, jacdifp, jacdifr, pos1, b1, 0, NV,
                         b1 < b2 ? 0 : m->body_dofnum[b2]);

      // second body
      MjuJacSparseSimple(m, d, jacdifp, jacdifr, pos2, b2, 1, NV,
                         b2 < b1 ? 0 : m->body_dofnum[b1]);
    }

    // regular processing
    else {
      // Jacobians
      MjuJacSparse(m, d, jac1p, jac1r, pos1, b1, NV, chain);
      MjuJacSparse(m, d, jac2p, jac2r, pos2, b2, NV, chain);

      // differences
      if (jacdifp) {
        mju_sub(jacdifp, jac2p, jac1p, 3 * NV);
      }
      if (jacdifr) {
        mju_sub(jacdifr, jac2r, jac1r, 3 * NV);
      }
    }
  }

  // dense case
  else {
    // Jacobians
    mj_jac(m, d, jac1p, jac1r, pos1, b1);
    mj_jac(m, d, jac2p, jac2r, pos2, b2);

    // differences
    if (jacdifp) {
      mju_sub(jacdifp, jac2p, jac1p, 3 * NV);
    }
    if (jacdifr) {
      mju_sub(jacdifr, jac2r, jac1r, 3 * NV);
    }
  }

  return NV;
}

// https://eigen.tuxfamily.org/dox/group__DenseDecompositionBenchmark.html
#define MJPC_USE_QR_INVERSE_MATRIX (0)
/* https://www.naukri.com/code360/library/understanding-svd-decomposition
 * JacobiSVD: For small matrices, two-sided Jacobi iterations are quickly implemented, but for bigger matrices, they take a very long time.
 * BDCSVD: Applying an upper-bidiagonalization that is still quick for large problems on top of a recursive divide-and-conquer approach.
 * -> Divide-and-conquer diagonalizes the input matrix after first reducing it to bi-diagonal form using class UpperBidiagonalization.
 */
#define MJPC_USE_JACOBI_SVD_INVERSE_MATRIX (1)
#define MJPC_USE_BDC_SVD_INVERSE_MATRIX (!MJPC_USE_QR_INVERSE_MATRIX && !MJPC_USE_JACOBI_SVD_INVERSE_MATRIX)

#if MJPC_USE_QR_INVERSE_MATRIX
/// Convenience method for pseudo-inverse
template <int i, int j, typename TMatrix = Eigen::Matrix<double, i, j>>
static inline TMatrix pinv(const Eigen::Matrix<double, i, j>& M) {
  return (M.completeOrthogonalDecomposition().pseudoInverse());
}
#elif MJPC_USE_JACOBI_SVD_INVERSE_MATRIX
// https://eigen.tuxfamily.org/dox/group__LeastSquares.html
// https://gist.github.com/javidcf/25066cf85e71105d57b6
template <int i, int j, typename TMatrix = Eigen::Matrix<double, i, j>>
static inline TMatrix Pinv(const Eigen::Matrix<double, i, j>& M,
                           double epsilon = std::numeric_limits<double>::epsilon()) {
#if 1
  Eigen::JacobiSVD<TMatrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  // For a non-square matrix
  // Eigen::JacobiSVD<TMatrix> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  double tolerance = epsilon * std::max(M.cols(), M.rows()) * svd.singularValues().array().abs()(0);
  return svd.matrixV() * (svd.singularValues().array().abs() > tolerance).
                         select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal()
         * svd.matrixU().adjoint();
#else
    // Ref: https://github.com/dqrobotics/cpp/blob/master/src/utils/DQ_LinearAlgebra.cpp
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
#elif MJPC_USE_BDC_SVD_INVERSE_MATRIX
// https://gist.github.com/pshriwise/67c2ae78e5db3831da38390a8b2a209f
template <int i, int j, typename TMatrix = Eigen::Matrix<double, i, j>>
static inline TMatrix pinv(const Eigen::Matrix<double, i, j>& M,
                           double epsilon = std::numeric_limits<double>::epsilon()) {
  Eigen::BDCSVD<TMatrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
  svd.setThreshold(epsilon * std::max(M.cols(), M.rows()));
  Eigen::Index rank = svd.rank();
  TMatrix tmp = svd.matrixU().leftCols(rank).adjoint();
  tmp = svd.singularValues().head(rank).asDiagonal().inverse() * tmp;
  return svd.matrixV().leftCols(rank) * tmp;
}
#endif

static inline Eigen::MatrixXd RobustInv(const Eigen::MatrixXd& M, const double alpha = 0.001) {
  auto Mt = M;
  Mt.transposeInPlace();
  return Mt * DQ_robotics::pinv(M * Mt + alpha * Eigen::MatrixXd::Identity(M.rows(), M.rows()));
}
}
