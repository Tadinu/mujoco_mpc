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

template <typename... TVariant>
using CasadiNamedVariantPair = std::pair<std::string, CasadiVariant<TVariant...>>;
template <typename... TVariant>
using CasadiNamedMap = std::map<std::string, CasadiVariant<TVariant...>>;
using CasadiDoubleScalarMap = CasadiNamedMap<double, std::vector<double>>;
using CasadiNamedAnyMap = std::map<std::string, std::any>;

template <typename... TVariant>
using CasadiVariantVector = std::vector<CasadiVariant<TVariant...>>;

static constexpr auto CASADI_INT_MIN = std::numeric_limits<casadi_int>::min();
static constexpr auto CASADI_INT_MAX = std::numeric_limits<casadi_int>::max();

#define CASADI_PRINT(...) mjpc_casadi::print(__VA_ARGS__)
#define CASADI_PRINTDB(...) mjpc_casadi::printdb(__VA_ARGS__)

namespace mjpc_casadi {
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

// ANY -------------------------------------------------------------------------------------------------------
//
template <typename T, typename... Types>
struct is_any_type : std::disjunction<std::is_same<T, Types>...> {};

template <typename T, typename... Types>
static constexpr bool is_any() {
  return (std::is_same_v<T, Types> || ...);
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
static std::any get_variant_value_any(const CasadiVariant<TArgs...>& var) {
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

template <typename T>
static constexpr bool is_convertible_to_casx() {
  return is_any<T, int, double, std::vector<int>, std::vector<double>, std::vector<std::vector<double>>,
                CaSX>();
}

template <typename... TArgs>
static bool variant_to_casx(const CasadiVariant<TArgs...>& var, CaSX& out) {
  bool res = false;
  (
      [&]() {
        // std::cout << typeid(TArgs).name() << std::endl;
        if constexpr (is_convertible_to_casx<TArgs>()) {
          TArgs val;
          if (get_variant_value2<TArgs>(var, val)) {
            out = CaSX(val);
            res = true;
          }
        }
      }(),
      ...);
  return res;
}

// PRINT -----------------------------------------------------------------------------------------------------
//
template <typename... TArgs>
static void print(TArgs&&... var) {
  ((std::cout << var << " "), ...) << std::endl;
}

template <typename... TArgs>
static void printdb(TArgs&&... var) {
#if CASADI_DEBUG
  print(std::forward<TArgs>(var)...);
#endif
}

template <typename... TArgs>
static void print_variant(const CasadiVariant<TArgs...>& var, const std::string& var_name = "") {
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
          } else if constexpr (std::is_same_v<TArgs, CaSX>) {
            std::cout << var_value << ": " << var_value.size() << std::endl;
          } else {
            std::cout << var_value << std::endl;
          }
        }
      }(),
      ...);
}

template <typename... TArgs>
static void print_named_map(const CasadiNamedMap<TArgs...>& vars, const char* label = nullptr) {
  if (label) print(label);
  for (const auto& [name, var] : vars) {
    print_variant(var, name);
  }
  print("----------------");
}

template <typename... TArgs>
static void print_named_mapdb(const CasadiNamedMap<TArgs...>& vars, const char* label = nullptr) {
#if CASADI_DEBUG
  print_named_map(vars, label);
#endif
}

template <typename TArg, typename TMap = std::map<std::string, TArg>>
static void print_named_map2(const TMap& map, const char* label = nullptr) {
  if (label) print(label);
  for (const auto& [name, val] : map) {
    print(name, ":", val);
  }
  print("----------------");
}

template <typename TArg, typename TMap = std::map<std::string, TArg>>
static void print_named_map2db(const TMap& map, const char* label = nullptr) {
#if CASADI_DEBUG
  print_named_map2<TArg>(map, label);
#endif
}

// CASX ------------------------------------------------------------------------------------------------------
//
static bool is_casx_sparse(const CaSX& expr) { return CaSX::symvar(expr).empty(); }

#if 0
// NOTE: Not all symbolic expression go through this parsing function!
static CaSXDict parse_symbolic_casx(const CaSX& expr, const std::vector<std::string>& var_names) {
  CaSXDict out_vars_dict;
  CASADI_PRINT("PARSE SYMBOLIC VARS OUTPUT:");
  for (const auto& var : CaSX::symvar(expr)) {
    if (has_collection_element(var_names, var.name())) {
      CASADI_PRINT(var.name(), var);
      out_vars_dict.insert_or_assign(var.name(), var);
    }
  }
  return out_vars_dict;
}
#endif

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
    CASADI_PRINT("Operation invalid", "Different dimensions: " + std::to_string(a.x().size().first) + "x" +
                                          std::to_string(a.x().size().second) + "vs. " +
                                          std::to_string(b.x().size().first) + "x" +
                                          std::to_string(b.x().size().second));
    return false;
  }

  if (!CaSX::is_equal(a.x(), b.x())) {
    CASADI_PRINT("Operation invalid", "Different values: " + a.x().get_str() + " vs. " + b.x().get_str());
    return false;
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
}  // namespace mjpc_casadi
