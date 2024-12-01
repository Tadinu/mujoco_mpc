#pragma once

#include <any>
#include <chrono>
#include <iostream>
#include <string>
#include <variant>

// MuJoCo
#include <mujoco/mujoco.h>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// Casadi
#include <casadi/casadi.hpp>

// MJPC
#include "mjpc/utils/mjpc_core_util.h"

#define CASADI_DEBUG (1)

using CaSX = casadi::SX;
using CaMX = casadi::MX;
using CaSXDict = casadi::SXDict;
using CaSXPair = std::pair<std::string, CaSX>;
using CaSXVector = casadi::SXVector;

using CaDM = casadi::DM;
using CaDMVector = casadi::DMVector;

using CaElement = casadi::SXElem;
using CaDouble = casadi::Matrix<double>;
using CaSlice = casadi::Slice;
using CaFunction = casadi::Function;

template <typename... TVariant>
using CasadiVariant = std::variant<std::monostate, TVariant...>;
using CasadiArg = CasadiVariant<CaSX, int, double, std::string, std::vector<int>, std::vector<double>,
                                std::map<int, double>>;
using CasadiArgMap = std::map<std::string, CasadiArg>;

static constexpr auto CASADI_INT_MIN = std::numeric_limits<casadi_int>::min();
static constexpr auto CASADI_INT_MAX = std::numeric_limits<casadi_int>::max();

#define CASADI_PRINT(...) mjpc::print(__VA_ARGS__)
#define CASADI_PRINTDB(...) mjpc::printdb(__VA_ARGS__)

namespace mjpc {
template <typename TScalar>
static CaSX CASX_IDENTITY(const TScalar size) {
  return CaSX::eye(size); /*with structural zeros*/  //+ CaSX::zeros(size, size); /*with scalar zeros*/
}
static CaSX CASX_TRANSF_IDENTITY = CASX_IDENTITY(4);

static CaSX CASX_3D_ZERO = CaSX::zeros(3);
static CaSX CASX_POSITION_ZERO = CaSX::zeros(3);
static CaSX CASX_ORIENTATION_ZERO = CaSX(std::vector<double>{1.0, 0.0, 0.0, 0.0});
static CaSX CASX_UNIT_X = CaSX({1, 0, 0});
static CaSX CASX_UNIT_Y = CaSX({0, 1, 0});
static CaSX CASX_UNIT_Z = CaSX({0, 0, 1});

template <typename TScalar, typename TCasadi = CaSX>
static TScalar squared_norm(const TCasadi& x) {
  return TScalar(TCasadi::norm_2(TCasadi::sq(x)).scalar());
}

template <typename TScalar, typename TCasadi = CaSX>
static TScalar norm_squared(const TCasadi& x) {
  return TScalar(TCasadi::pow(TCasadi::norm_2(x), 2).scalar());
}

template <typename T>
static constexpr bool is_convertible_to_casx() {
  return mjpc::is_any<T, int, double, std::vector<int>, std::vector<double>, std::vector<std::vector<double>>,
                      CaSX>();
}

template <typename... TArgs>
static CaSXDict get_casx_dict(const MjpcNamedMap<TArgs...>& vars) {
  CaSXDict res;
  for (const auto& item_var : vars) {
    (
        [&]() {
          const auto& name = item_var.first;
          const auto& var = item_var.second;
          TArgs val;
          if (get_variant_value2<TArgs>(var, val)) {
            res.insert_or_assign(name, CaSX(val));
          }
        }(),
        ...);
  }
  return res;
}

template <typename... TArgs>
static bool variant_to_casx(const MjpcVariant<TArgs...>& var, CaSX& out) {
  bool res = false;
  (
      [&]() {
        // std::cout << typeid(TArgs).name() << std::endl;
        if constexpr (is_convertible_to_casx<TArgs>()) {
          TArgs val;
          if (mjpc::get_variant_value2<TArgs>(var, val)) {
            out = CaSX(val);
            res = true;
          }
        }
      }(),
      ...);
  return res;
}

template <typename... TArgs>
static void print_casadi_variant(const MjpcVariant<TArgs...>& var, const std::string& var_name = "") {
  (
      [&]() {
        if (const auto* var_value_ptr = std::get_if<TArgs>(&var)) {
          const auto var_value = *var_value_ptr;
          if constexpr (std::is_same_v<TArgs, CaSX>) {
            std::cout << var_value << ": " << var_value.size() << std::endl;
          } else {
            mjpc::print_variant(var, var_name);
          }
        }
      }(),
      ...);
}

template <typename T>
static std::string join_casadi_str(const std::vector<T>& inputs, const std::string& delimiter = ",") {
  std::string result;
  for (const auto& str : inputs) {
    if constexpr (std::is_same_v<T, std::string>) {
      result += str;
    } else if constexpr (std::is_same_v<T, CaSX>) {
      result += str.get_str();
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

static bool is_casx_sparse(const CaSX& expr) { return CaSX::symvar(expr).empty(); }

// NOTE: Not all symbolic expression go through this parsing function!
static CaSXDict parse_symbolic_casx(const CaSX& expr, const std::vector<std::string>& var_names) {
  CaSXDict out_vars_dict;
  MJPC_PRINT("PARSE SYMBOLIC VARS OUTPUT:");
  for (const auto& var : CaSX::symvar(expr)) {
    if (mjpc::has_collection_element(var_names, var.name())) {
      MJPC_PRINT(var.name(), var);
      out_vars_dict.insert_or_assign(var.name(), var);
    }
  }
  return out_vars_dict;
}

static bool is_equal_SXPair(const CaSXPair& left, const CaSXPair& right) {
  return (left.first == right.first) && CaSX::is_equal(left.second, right.second);
}

static bool is_equal_SXDict(const CaSXDict& left, const CaSXDict& right) {
  return std::equal(left.begin(), left.end(), right.begin(), right.end(), is_equal_SXPair);
}

template <typename TIteratable>
static bool is_equal_itertable(const TIteratable& left, const TIteratable& right) {
  return std::equal(left.begin(), left.end(), right.begin(), right.end());
}

template <typename TGeometricComponent1, typename TGeometricComponent2>
static bool check_compatibility(const TGeometricComponent1& a, const TGeometricComponent2& b) {
  if (a.x().size() != b.x().size()) {
    throw MjpcError::customized("Operation invalid",
                                "Different dimensions: " + std::to_string(a.x().size().first) + "x" +
                                    std::to_string(a.x().size().second) + "vs. " +
                                    std::to_string(b.x().size().first) + "x" +
                                    std::to_string(b.x().size().second));
  }

  if (!CaSX::is_equal(a.x(), b.x())) {
    throw MjpcError::customized("Operation invalid",
                                "Different values: " + a.x().get_str() + " vs. " + b.x().get_str());
  }
  return true;
}

static CaSX get_casx(const CaSX& a, const std::vector<int>& filtering_indices, bool indx1 = false) {
  CaSX elem;
  a.get(elem, indx1, filtering_indices);
  return elem;
}

static CaSX get_casx(const CaSX& a, const casadi_int index, bool indx1 = false) {
  CaSX elem;
  a.get(elem, indx1, CaSlice(index));
  return elem;
}

static CaSX get_casx(const CaSX& a, const std::array<casadi_int, 2>& indices, bool indx1 = false) {
  CaSX elem;
  a.get(elem, indx1, CaSlice(indices[0], indices[1]));
  return elem;
}

static CaSX get_casx2(const CaSX& a, const casadi_int index_1, const casadi_int index_2, bool indx1 = false) {
  CaSX elem;
  a.get(elem, indx1, CaSlice(index_1), CaSlice(index_2));
  return elem;
}

static CaSX get_casx2(const CaSX& a, const std::array<casadi_int, 2> index_1, const casadi_int index_2,
                      bool indx1 = false) {
  CaSX elem;
  a.get(elem, indx1, CaSlice(index_1[0], index_1[1]), CaSlice(index_2));
  return elem;
}

static CaSX get_casx2(const CaSX& a, const std::array<casadi_int, 2> index_1,
                      const std::array<casadi_int, 2> index_2, bool indx1 = false) {
  CaSX elem;
  a.get(elem, indx1, CaSlice(index_1[0], index_1[1]), CaSlice(index_2[0], index_2[1]));
  return elem;
}

static void set_casx(CaSX& a, const casadi_int index, const CaSX& b, bool indx1 = false) {
  a.set(b, indx1, CaSlice(index));
}

static void set_casx(CaSX& a, const std::array<casadi_int, 2>& indices, const CaSX& b, bool indx1 = false) {
  a.set(b, indx1, CaSlice(indices[0], indices[1]));
}

static void set_casx2(CaSX& a, const casadi_int index_1, const casadi_int index_2, const CaSX& b,
                      bool indx1 = false) {
  a.set(b, indx1, CaSlice(index_1), CaSlice(index_2));
}

static void set_casx2(CaSX& a, const std::array<casadi_int, 2> index_1, const casadi_int index_2,
                      const CaSX& b, bool indx1 = false) {
  a.set(b, indx1, CaSlice(index_1[0], index_1[1]), CaSlice(index_2));
}

static void set_casx2(CaSX& a, const std::array<casadi_int, 2> index_1,
                      const std::array<casadi_int, 2> index_2, const CaSX& b, bool indx1 = false) {
  a.set(b, indx1, CaSlice(index_1[0], index_1[1]), CaSlice(index_2[0], index_2[1]));
}

// MUJOCO <-> CASADI -----------------------------------------------------------------------------------------
//
template <typename TCasadi = CaSX>
static TCasadi from_mjarray(const mjtNum* mj_data, int num) {
  TCasadi val = TCasadi::zeros(num);
  for (auto i = 0; i < num; ++i) {
    val(i) = mj_data[i];
  }
  return val;
}

template <typename TCasadi = CaSX>
static TCasadi from_mjpos(const mjtNum* mj_data) {
  return from_mjarray(mj_data, 3);
}

template <typename TCasadi = CaSX>
static TCasadi from_mjquat(const mjtNum* mj_data) {
  return from_mjarray(mj_data, 4);
}

template <typename TCasadi = CaSX>
static TCasadi from_mjvel(const mjtNum* mj_data) {
  return from_mjpos(mj_data);
}

template <typename TCasadi = CaSX>
static TCasadi from_mjacc(const mjtNum* mj_data) {
  return from_mjpos(mj_data);
}

// EIGEN <-> CASADI ------------------------------------------------------------------------------------------
//
template <typename TEigen = Eigen::Vector3d, typename TCasadi = CaSX>
static TCasadi from_eigen_vector(const TEigen& eigen_vector, int num) {
  TCasadi val = TCasadi::zeros(num);
  for (auto i = 0; i < num; ++i) {
    val(i) = eigen_vector[i];
  }
  return val;
}

template <typename TCasadi = CaSX>
static TCasadi from_eigen_vector3(const Eigen::Vector3d& eigen_vector) {
  return from_eigen_vector(eigen_vector, 3);
}

template <typename TEigen = Eigen::Vector3d, typename TCasadi = CaSX>
static TEigen to_eigen_vector(const TCasadi& ca_val, int num) {
  TEigen eigen_val;
  for (auto i = 0; i < num; ++i) {
    eigen_val[i] = (double)ca_val(i).scalar();
  }
  return eigen_val;
}

template <typename TCasadi = CaSX>
static Eigen::Vector3d to_eigen_vector3(const TCasadi& ca_val) {
  return to_eigen_vector<Eigen::Vector3d>(ca_val, 3);
}

template <typename TEigen = Eigen::Quaterniond, typename TCasadi = CaSX>
static TCasadi from_eigen_quat(const TEigen& eigen_quat) {
  return TCasadi({eigen_quat.w(), eigen_quat.x(), eigen_quat.y(), eigen_quat.z()});
}

template <typename TEigen = Eigen::Quaterniond, typename TCasadi = CaSX>
static TEigen to_eigen_quat(const TCasadi& ca_val) {
  return TEigen((double)ca_val(3).scalar(), (double)ca_val(0).scalar(), (double)ca_val(1).scalar(),
                (double)ca_val(2).scalar());
}

template <typename TCasadi = CaSX>
static void add_gaussian_noise(TCasadi& x) {
  // perturb all vars by gaussian noise
  static constexpr double mean = 0.;
  static constexpr double var = 0.01;

  static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  static std::default_random_engine generator(seed);
  static std::normal_distribution<double> distribution(mean, var);
  for (auto i = 0; i < x.size1(); ++i) {
    for (auto j = 0; j < x.size2(); ++i) {
      x(i, j) += std::abs(distribution(generator));
    }
  }
}
}  // namespace mjpc
