#pragma once

#include <any>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
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

// MJPC
#include "mjpc/core/mjpc_common.h"

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
struct is_any_type : std::disjunction<std::is_same<T, Types>...> {};

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
}  // namespace mjpc