#pragma once

#include <algorithm>
#include <any>
#include <casadi/casadi.hpp>
#include <regex>
#include <variant>
#include <vector>

// abseil
#include "absl/strings/ascii.h"
#include "absl/strings/str_split.h"

// fabrics
#include "mjpc/planners/fabrics/include/fab_common.h"

#define FAB_PRINT(...) fab_core::print(__VA_ARGS__)
#define FAB_PRINTDB(...) fab_core::printdb(__VA_ARGS__)

namespace fab_core {
template <typename... TArgs>
static void print(TArgs&&... var) {
  ((std::cout << var << " "), ...) << std::endl;
}

template <typename... TArgs>
static void printdb(TArgs&&... var) {
#if FAB_DEBUG
  print(std::forward<TArgs>(var)...);
#endif
}

template <typename... TArgs>
static void print_variant(const FabVariant<TArgs...>& var, const std::string& var_name = "") {
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
static void print_named_map(const FabNamedMap<TArgs...>& vars, const char* label = nullptr) {
  if (label) print(label);
  for (const auto& [name, var] : vars) {
    print_variant(var, name);
  }
  print("----------------");
}

template <typename... TArgs>
static void print_named_mapdb(const FabNamedMap<TArgs...>& vars, const char* label = nullptr) {
#if FAB_DEBUG
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
#if FAB_DEBUG
  print_named_map2<TArg>(map, label);
#endif
}

template <typename T, typename... Types>
struct is_any_type : std::disjunction<std::is_same<T, Types>...> {};

template <typename T, typename... Types>
constexpr bool is_any() {
  return (std::is_same_v<T, Types> || ...);
}

template <typename T>
constexpr bool is_convertible_to_casx() {
  return is_any<T, int, double, std::vector<int>, std::vector<double>, std::vector<std::vector<double>>,
                CaSX>();
}

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

template <typename T>
static std::string join(const std::vector<T>& inputs, const std::string& delimiter = ",") {
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
    } else if constexpr (std::is_same_v<TValue, CaSX>) {
      result += val.get_str();
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

template <typename TArg, typename... TArgs, typename = std::enable_if_t<(std::is_same_v<TArg, TArgs> || ...)>>
static const TArg* get_arg_value(const FabNamedMap<TArgs...>& kwargs, const char* arg_name) {
  if (auto it = kwargs.find(arg_name); it != std::end(kwargs)) {
    if (const auto* arg_value_ptr = std::get_if<TArg>(&it->second)) {
      return arg_value_ptr;
    }
    throw FabParamNotFoundError("Parameter [" + std::string(arg_name) + "] is not of " + typeid(TArg).name());
  }
  return nullptr;
}

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
static std::any get_variant_value_any(const FabVariant<TArgs...>& var) {
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
static FabNamedAnyMap get_named_any_map(const FabNamedMap<TArgs...>& vars) {
  FabNamedAnyMap res;
  for (const auto& [name, val] : vars) {
    res.insert_or_assign(name, get_variant_value_any<TArgs...>(val));
  }
  return res;
}

template <typename... TArgs>
static CaSXDict get_casx_dict(const FabNamedMap<TArgs...>& vars) {
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
static bool variant_to_casx(const FabVariant<TArgs...>& var, CaSX& out) {
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

template <typename T>
static std::vector<T> tokenize(const std::string& text, const std::string& delimiter = " ") {
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
    std::transform(strings.begin(), strings.end(), std::back_inserter(results),
                   [](auto& token) { return absl::StripAsciiWhitespace(token); });
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

static std::string task_function_name(const std::string& task_name, bool is_goal_fixed, bool are_obst_fixed) {
  return task_name + (is_goal_fixed ? "_static" : "_dynamic") + "_goal" +
         (are_obst_fixed ? "_static" : "_dynamic") + "_obst";
}

// -----------------------------------------------------------------------------------------------------------
// CASADI UTILS ==
//
static bool is_casx_sparse(const CaSX& expr) { return CaSX::symvar(expr).empty(); }

static CaSX casx_sym(const std::string& name, const casadi_int dim = 1) { return CaSX::sym(name, dim); }

// NOTE: Not all symbolic expression go through this parsing function!
static CaSXDict parse_symbolic_casx(const CaSX& expr, const std::vector<std::string>& var_names) {
  CaSXDict out_vars_dict;
  FAB_PRINT("PARSE SYMBOLIC VARS OUTPUT:");
  for (const auto& var : CaSX::symvar(expr)) {
    if (has_collection_element(var_names, var.name())) {
      FAB_PRINT(var.name(), var);
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
    throw FabError::customized("Operation invalid",
                               "Different dimensions: " + std::to_string(a.x().size().first) + "x" +
                                   std::to_string(a.x().size().second) + "vs. " +
                                   std::to_string(b.x().size().first) + "x" +
                                   std::to_string(b.x().size().second));
  }

  if (!CaSX::is_equal(a.x(), b.x())) {
    throw FabError::customized("Operation invalid",
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
}  // namespace fab_core
