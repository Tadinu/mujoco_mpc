#pragma once

#include <any>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <numeric>
#include <variant>

// absl
#include <absl/strings/str_join.h>

// MuJoCo
#include <mujoco/mujoco.h>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// abseil
#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"

// DQ-Robotics
#include <dqrobotics/utils/DQ_LinearAlgebra.h>

// MJPC
#include "mjpc/core/mjpc_common.h"
#include "mjpc/planners/bimanual/dq_franka_robot.h"

namespace mjpc {
// PRINT -----------------------------------------------------------------------------------------------------
//
#define MJPC_PRINT(...) mjpc::print(__VA_ARGS__)
#define MJPC_PRINTDB(...) mjpc::printdb(__VA_ARGS__)

template <typename... TArgs>
static void print(const TArgs&... var) {
  ((std::cout << var << " "), ...) << std::endl;
}

template <typename... TArgs>
static void printdb(const TArgs&... var) {
#if MJPC_DEBUG
  print(var);
#endif
}

template <typename... TArgs>
static void print_variant(const MjpcVariant<TArgs...>& var, const std::string& var_name = "") {
  (
    [&]() {
      if (const auto* var_value_ptr = std::get_if<TArgs>(&var)) {
        const auto var_value = *var_value_ptr;
        if (!var_name.empty()) {
          std::cout << var_name << ": ";
        }
        if constexpr (std::is_same_v<TArgs, std::any>) {
          if (var_value.has_value()) {
            try {
              std::cout << std::any_cast<std::string>(var_value) << std::endl;
            } catch (const std::bad_any_cast& e) {
            }
          }
        } else {
          std::cout << var_value << std::endl;
        }
      }
    }(),
    ...);
}

template <typename... TArgs>
static void print_named_map(const MjpcNamedMap<TArgs...>& vars, const char* label = nullptr) {
  if (label) print(label);
  for (const auto& [name, var] : vars) {
    print_variant(var, name);
  }
  print("----------------");
}

template <typename... TArgs>
static void print_named_mapdb(const MjpcNamedMap<TArgs...>& vars, const char* label = nullptr) {
#if MJPC_DEBUG
  print_named_map(vars, label);
#endif
}

template <typename TArg, typename TMap = std::map<std::string, TArg>>
static void print_named_map2(const TMap& map, const char* label = nullptr) {
  if (label) print(label);
  for (const auto& [name, val] : map) {
    if constexpr (std::is_same_v<TArg, std::vector<std::string>>) {
      print(name, ":", absl::StrJoin(val, ","));
    } else if constexpr (std::is_same_v<TArg, std::pair<std::string, std::string>>) {
      print(name, ": [", val.first, val.second, "]");
    } else if constexpr (std::is_same_v<TArg, std::vector<std::pair<std::string, std::string>>>) {
      std::vector<std::string> pair_list;
      for (const auto& [first, second] : val) {
        pair_list.push_back("[" + first + "," + second + "]");
      }
      print(name, ":", absl::StrJoin(pair_list, ","));
    } else {
      print(name, ":", val);
    }
  }
  print("----------------");
}

template <typename TArg, typename TMap = std::map<std::string, TArg>>
static void print_named_map2db(const TMap& map, const char* label = nullptr) {
#if MJPC_DEBUG
  print_named_map2<TArg>(map, label);
#endif
}

// ANY -------------------------------------------------------------------------------------------------------
//
template <typename T, typename... Types>
struct is_any_type : std::disjunction<std::is_same<T, Types>...> {
};

template <typename T, typename... Types>
static constexpr bool is_any() {
  return (std::is_same_v<T, Types> || ...);
}

// ARG -------------------------------------------------------------------------------------------------------
//
template <typename TArg, typename... TArgs, typename = std::enable_if_t<(std::is_same_v<TArg, TArgs> || ...)>>
static const TArg* get_arg_value(const MjpcNamedMap<TArgs...>& kwargs, const char* arg_name) {
  if (auto it = kwargs.find(arg_name); it != std::end(kwargs)) {
    if (const auto* arg_value_ptr = std::get_if<TArg>(&it->second)) {
      return arg_value_ptr;
    }
    throw MjpcParamNotFoundError("Parameter [" + std::string(arg_name) + "] is not of " +
                                 typeid(TArg).name());
  }
  return nullptr;
}

// VARIANT ---------------------------------------------------------------------------------------------------
//
template <typename T, typename TVariant>
static T get_variant_value(const TVariant& variant) {
  if (const auto* value_ptr = std::get_if<T>(&variant)) {
    return *value_ptr;
  }
  return T();
}

template <typename T, typename TVariant>
static bool get_variant_value2(const TVariant& variant, T& out) {
  if (const auto* value_ptr = std::get_if<T>(&variant)) {
    out = *value_ptr;
    return true;
  }
  return false;
}

template <typename... TArgs>
static std::any get_variant_value_any(const MjpcVariant<TArgs...>& var) {
  std::any res;
  (
    [&]() {
      if (const auto* value_ptr = std::get_if<TArgs>(&var)) {
        res = std::any(*value_ptr);
      }
    }(),
    ...);
  return res;
}

template <typename T>
static bool get_any_value(const std::any& any_var, T& out_val) {
  try {
    out_val = std::any_cast<T>(any_var);
    return true;
  } catch (const std::bad_any_cast& e) {
    return false;
  }
}

template <typename... TArgs>
static MjpcNamedAnyMap get_named_any_map(const MjpcNamedMap<TArgs...>& vars) {
  MjpcNamedAnyMap res;
  for (const auto& [name, val] : vars) {
    res.insert_or_assign(name, get_variant_value_any<TArgs...>(val));
  }
  return res;
}

// COLLECTION ------------------------------------------------------------------------------------------------
//
template <typename TMap>
static std::vector<std::string> get_map_keys(const TMap& variants) {
#if 1
  std::vector<std::string> names;
  std::transform(variants.begin(), variants.end(), std::back_inserter(names),
                 [](auto& variant) { return variant.first; });
  return names;
#else
  // c++20
  auto kv = std::views::keys(variants);
  return {kv.begin(), kv.end()};
#endif
}

template <typename TValue, typename TMap>
static std::vector<TValue> get_map_values(const TMap& variants) {
  std::vector<TValue> values;
  std::transform(variants.begin(), variants.end(), std::back_inserter(values),
                 [](auto& variant) { return variant.second; });
  return values;
}

template <typename T, typename TCollection>
static bool has_collection_element(const TCollection& collection, const T& elem) {
  return std::find(collection.begin(), collection.end(), elem) != collection.end();
}

template <typename TCollection, class TPredicate>
static bool has_collection_element_if(const TCollection& collection, const TPredicate& pred) {
  return std::find_if(collection.begin(), collection.end(), pred) != collection.end();
}

template <typename TCollection>
static TCollection get_subcollection(const TCollection& collection, const std::vector<int>& indices = {}) {
  if (indices.empty()) {
    return collection;
  }
  TCollection subcollection;
  std::transform(indices.begin(), indices.end(), std::back_inserter(subcollection),
                 [&collection](int index) { return collection.at(index); });
  return subcollection;
}

template <typename T>
static std::string join(const std::vector<T>& inputs, const std::string& delimiter = ",") {
  std::string result;
  for (const auto& str : inputs) {
    if constexpr (std::is_same_v<T, std::string>) {
      result += str;
    } else {
      result += std::to_string(str);
    }
    result += delimiter;
  }
  if (const auto pos = result.find_last_of(delimiter); pos != std::string::npos) {
    result.erase(pos);
  }
  return result;
}

template <typename TKey, typename TValue>
static std::string join(const std::map<TKey, TValue>& inputs, const std::string& delimiter = ",") {
  std::string result;
  for (const auto& [key, val] : inputs) {
    result += "{";

    // Key
    if constexpr (std::is_same_v<TKey, std::string>) {
      result += key;
    } else {
      result += std::to_string(key);
    }
    result += ",";

    // Value
    if constexpr (std::is_same_v<TValue, std::string>) {
      result += val;
    } else {
      result += std::to_string(val);
    }
    result += "}" + delimiter + "\n";
  }
  if (const auto pos = result.find_last_of(delimiter); pos != std::string::npos) {
    result.erase(pos);
  }
  return result;
}

template <typename T>
static std::vector<T> tokenize(const std::string& text, const std::string& delimiter = " ",
                               bool allow_whitespace = false) {
#if 0
  std::vector<T> results;
  std::stringstream size_stream(text);
  bool has_token = true;
  do {
    std::string token;
    has_token = bool(getline(size_stream, token, delimiter[0]));
    if constexpr (std::is_same_v<T, std::string>) {
      results.emplace_back(std::move(token));
    } else if constexpr (std::is_scalar_v<T>) {
      results.push_back(std::stoi(token));
    } else if constexpr (std::is_floating_point_v<T>) {
      results.push_back(std::stod(token));
    }
  } while (has_token);
  return results;
#else
  std::vector<T> results;
  std::vector<std::string> strings = absl::StrSplit(text, delimiter);
  if constexpr (std::is_same_v<T, std::string>) {
    if (allow_whitespace) {
      results = std::move(strings);
    } else {
      std::transform(strings.begin(), strings.end(), std::back_inserter(results),
                     [](auto& token) { return std::string(absl::StripAsciiWhitespace(token)); });
    }
  } else {
    std::transform(strings.begin(), strings.end(), std::back_inserter(results), [](auto& token) {
      if constexpr (std::is_floating_point_v<T>) {
        return std::stod(token);
      } else if constexpr (std::is_scalar_v<T>) {
        return std::stoi(token);
      }
    });
  }
  return results;
#endif
}

// CONVERSION UTILS ---------
// EIGEN
//
static void SetEigenVector(Eigen::VectorXd& vec, const std::vector<int>& indices,
                           const std::vector<double>& values) {
  assert(indices.size() == values.size());
  for (auto i = 0; i < indices.size(); ++i) {
    vec(indices[i]) = values[i];
  }
}

static void ResetEigenVector(Eigen::VectorXd& vec, const std::vector<int>& indices) {
#if 1
  vec(Eigen::Map<const Eigen::VectorXi>(indices.data(), indices.size())).setZero();
#else
  for (auto i = 0; i < indices.size(); ++i) {
    vec(indices[i]) = 0.0;
  }
#endif
}

static Eigen::VectorXd PosToEigen(const mjtNum* pos, int n = 3) {
  return Eigen::Map<const Eigen::VectorXd>(pos, n);
}

static const mjtNum* PosFromEigen(const Eigen::VectorXd& pos) {
  return pos.data();
}

static std::vector<mjtNum> VectorFromEigen(const Eigen::VectorXd& pos) {
  return std::vector(pos.data(), pos.data() + pos.size());
}

static Eigen::Quaterniond QuatToEigen(const mjtNum quat[4]) {
  // NOTE: Don't use Quaterniond(Scalar* data) which assigns data directly to m_coeffs, which is stored in XYZW
  return Eigen::Quaterniond(/*W*/quat[0], /*X*/quat[1], /*Y*/quat[2], /*Z*/quat[3]);
}

static mjtNum* QuatFromEigen(const Eigen::Quaterniond& equat) {
  static mjtNum quat[4];
  quat[0] = equat.w();
  quat[1] = equat.x();
  quat[2] = equat.y();
  quat[3] = equat.z();
  return quat;
}

static Eigen::MatrixXd StackEigenMatrices(const std::vector<Eigen::MatrixXd>& matrices, bool horizontally) {
  const auto total_num = std::accumulate(matrices.begin(), matrices.end(), 0,
                                         [horizontally](const int count, const Eigen::MatrixXd& item) {
                                           return count + (horizontally ? item.cols() : item.rows());
                                         });
  Eigen::MatrixXd stacked(horizontally ? matrices[0].rows() : total_num,
                          horizontally ? total_num : matrices[0].cols());
  int offset = 0;
  for (const auto& m : matrices) {
    const int num = horizontally ? m.cols() : m.rows();
    if (horizontally) {
      stacked.middleCols(offset, num) = m;
    } else {
      stacked.middleRows(offset, num) = m;
    }
    offset += num;
  }
  return stacked;
}

static Eigen::VectorXd JoinEigenVectors(const std::vector<Eigen::VectorXd>& vectors) {
  Eigen::VectorXd stacked(std::accumulate(vectors.begin(), vectors.end(), 0,
                                          [](const int count, const Eigen::VectorXd& item) {
                                            return count + item.size();
                                          }));
  int offset = 0;
  for (const auto& v : vectors) {
    const int num = v.size();
    stacked.segment(offset, num) = v;
    offset += num;
  }
  return stacked;
}

// MATH ---------------------
//
// Ref: https://github.com/google-deepmind/mujoco/blob/main/src/user/user_util.h
// convert global to local axis relative to given frame
static void mjpc_localaxis(double* al, const double* ag, const double* quat) {
  double mat[9];
  double qneg[4] = {quat[0], -quat[1], -quat[2], -quat[3]};
  mju_quat2Mat(mat, qneg);
  mju_mulMatVec3(al, ag, mat);
}

// Ref: mj_local2Global()
// convert global to local position relative to given frame
static void mjpc_localpos(double* pl, const double* pg, const double* pos, const double* quat) {
  double a[3] = {pg[0] - pos[0], pg[1] - pos[1], pg[2] - pos[2]};
  mjpc_localaxis(pl, a, quat);
}

// compute quaternion rotation from parent to child
static void mjpc_localquat(double* local, const double* child, const double* parent) {
  double pneg[4] = {parent[0], -parent[1], -parent[2], -parent[3]};
  mju_mulQuat(local, pneg, child);
}

// Ref: mj_fullM
// Convert sparse inertia matrix M into full (i.e. dense) matrix.
static void mjpc_fullMatrix(const mjModel* m, mjtNum* dst, const mjtNum* M /* inertial matrix: qM*/,
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
static int mjpc_bodyChain(const mjModel* m, int* chain, int body, int base_body = 0) {
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
    while (da >= base_body) {
      chain[NV++] = da;
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


// Ref: [engine_support.h] - mj_jacSparse()
static void mjpc_jacSparse(const mjModel* m, const mjData* d,
                           mjtNum* jacp, mjtNum* jacr, const mjtNum* point, int body,
                           int NV, const int* chain) {
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
  while (da >= 0) {
    // find chain index for this dof
    while (ci >= 0 && chain[ci] > da) {
      ci--;
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

// https://eigen.tuxfamily.org/dox/group__DenseDecompositionBenchmark.html
#define MJPC_USE_QR_INVERSE_MATRIX (1)
/* https://www.naukri.com/code360/library/understanding-svd-decomposition
 * JacobiSVD: For small matrices, two-sided Jacobi iterations are quickly implemented, but for bigger matrices, they take a very long time.
 * BDCSVD: Applying an upper-bidiagonalization that is still quick for large problems on top of a recursive divide-and-conquer approach.
 * -> Divide-and-conquer diagonalizes the input matrix after first reducing it to bi-diagonal form using class UpperBidiagonalization.
 */
#define MJPC_USE_JACOBI_SVD_INVERSE_MATRIX (0)
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
  template<int i, int j, typename TMatrix = Eigen::Matrix<double, i, j>>
  static inline TMatrix pinv(const Eigen::Matrix<double, i, j> &M,
                             double epsilon = std::numeric_limits<double>::epsilon()) {
#if 1
    Eigen::JacobiSVD<TMatrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // For a non-square matrix
    // Eigen::JacobiSVD<TMatrix> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = epsilon * std::max(M.cols(), M.rows()) * svd.singularValues().array().abs()(0);
    return svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
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
  template<int i, int j, typename TMatrix = Eigen::Matrix<double, i, j>>
  static inline TMatrix pinv(const Eigen::Matrix<double, i, j> &M,
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

static inline Eigen::MatrixXd robust_inv(const Eigen::MatrixXd& M, const double alpha = 0.001) {
  auto Mt = M;
  Mt.transposeInPlace();
  return Mt * DQ_robotics::pinv(M * Mt + alpha * Eigen::MatrixXd::Identity(M.rows(), M.rows()));
}
} // namespace mjpc