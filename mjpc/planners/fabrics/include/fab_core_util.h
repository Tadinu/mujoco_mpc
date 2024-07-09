#pragma once

#include <algorithm>
#include <casadi/casadi.hpp>
#include <regex>
#include <vector>

#include "mjpc/planners/fabrics/include/fab_common.h"

namespace fab_core {
template <typename TMap>
static std::vector<std::string> get_map_keys(const TMap& variants) {
  std::vector<std::string> names;
  std::transform(variants.begin(), variants.end(), std::back_inserter(names),
                 [](auto& variant) { return variant.first; });
  return names;
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

static std::string join(const std::vector<std::string> inputs, const std::string& token) {
  std::string result;
  for (const auto& str : inputs) {
    result += str + token;
  }
  if (const auto pos = result.find_last_of(token); pos != std::string::npos) {
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
    throw FabParamNotFoundError(std::string("Parameter [") + arg_name + "] is not of " + typeid(TArg).name());
  }
  return nullptr;
}

template <typename T>
static std::vector<T> tokenize(const std::string& text, const std::string& token) {
  std::vector<T> results;
  std::stringstream size_stream(text);
  bool has_token = true;
  do {
    std::string token;
    has_token = bool(getline(size_stream, token, ' '));
    if constexpr (std::is_same_v<T, std::string>) {
      results.emplace_back(std::move(token));
    } else if constexpr (std::is_scalar_v<T>) {
      results.push_back(std::stoi(token));
    } else if constexpr (std::is_floating_point_v<T>) {
      results.push_back(std::stod(token));
    }
  } while (has_token);
  return results;
}

// -----------------------------------------------------------------------------------------------------------
// CASADI UTILS ==
//
static bool is_casx_sparse(const CaSX& expr) { return !CaSX::symvar(expr).empty(); }

static CaSX casx_sym(const std::string& name) { return CaSX::sym(name, 1); }

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
    throw FabError::customized("Operation invalid", std::string("Different dimensions: ") +
                                                        std::to_string(a.x().size().first) + "x" +
                                                        std::to_string(a.x().size().second) + "vs. " +
                                                        std::to_string(b.x().size().first) + "x" +
                                                        std::to_string(b.x().size().second));
  }

  if (!CaSX::is_equal(a.x(), b.x())) {
    throw FabError::customized(
        "Operation invalid", std::string("Different values: ") + a.x().get_str() + " vs. " + b.x().get_str());
  }
  return true;
}

static CaSX outer_product(const CaSX& a, const CaSX& b) {
  const auto m = a.size().first;
  const auto A = CaSX(CaSX::repmat(a.T(), m)).T();
  const auto B = CaSX::repmat(b.T(), m);
  return CaSX::times(A, B);
}

static CaSX get_casx(const CaSX& a, const std::vector<casadi_int>& filtering_indices) {
  CaSX elem;
  a.get(elem, false, filtering_indices);
  return elem;
}

static CaSX get_casx(const CaSX& a, const casadi_int index) {
  CaSX elem;
  a.get(elem, false, CaSlice(index));
  return elem;
}

static CaSX get_casx(const CaSX& a, const casadi_int start_idx, const casadi_int end_idx) {
  CaSX elem;
  a.get(elem, false, CaSlice(start_idx, end_idx));
  return elem;
}

static CaSX get_casx2(const CaSX& a, const casadi_int index_1, const casadi_int index_2) {
  CaSX elem;
  a.get(elem, false, CaSlice(index_1), CaSlice(index_2));
  return elem;
}

static CaSX get_casx2(const CaSX& a, const std::array<casadi_int, 2> index_1, const casadi_int index_2) {
  CaSX elem;
  a.get(elem, false, CaSlice(index_1[0], index_1[1]), CaSlice(index_2));
  return elem;
}

static CaSX get_casx2(const CaSX& a, const std::array<casadi_int, 2> index_1,
                      const std::array<casadi_int, 2> index_2) {
  CaSX elem;
  a.get(elem, false, CaSlice(index_1[0], index_1[1]), CaSlice(index_2[0], index_2[1]));
  return elem;
}

static void set_casx(CaSX& a, const casadi_int index, const CaSX& b) { a.set(b, false, CaSlice(index)); }
static void set_casx(CaSX& a, const casadi_int start_idx, const casadi_int end_idx, const CaSX& b) {
  a.set(b, false, CaSlice(start_idx, end_idx));
}
static void set_casx2(CaSX& a, const casadi_int index_1, const casadi_int index_2, const CaSX& b) {
  a.set(b, false, CaSlice(index_1), CaSlice(index_2));
}
static void set_casx2(CaSX& a, const std::array<casadi_int, 2> index_1, const casadi_int index_2,
                      const CaSX& b) {
  a.set(b, false, CaSlice(index_1[0], index_1[1]), CaSlice(index_2));
}
static void set_casx2(CaSX& a, const std::array<casadi_int, 2> index_1,
                      const std::array<casadi_int, 2> index_2, const CaSX& b) {
  a.set(b, false, CaSlice(index_1[0], index_1[1]), CaSlice(index_2[0], index_2[1]));
}
};  // namespace fab_core
