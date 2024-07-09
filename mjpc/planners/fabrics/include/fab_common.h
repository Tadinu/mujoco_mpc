#pragma once
// Dynamic Optimization Fabrics for Motion Generation
// https://arxiv.org/abs/2205.08454
// https://github.com/tud-amr/fabrics

#include <casadi/casadi.hpp>
#include <variant>

using CaSX = casadi::SX;
using CaMX = casadi::MX;
using CaSXDict = casadi::SXDict;
using CaSXPair = std::pair<std::string, CaSX>;
using CaSXVector = casadi::SXVector;
using CaElement = casadi::SXElem;
using CaDouble = casadi::Matrix<double>;
using CaSlice = casadi::Slice;
using CaFunction = casadi::Function;

template <typename... TVariant>
using FabVariant = std::variant<std::monostate, TVariant...>;

template <typename... TVariant>
using FabNamedVariantPair = std::pair<std::string, FabVariant<TVariant...>>;
template <typename... TVariant>
using FabNamedMap = std::map<std::string, FabVariant<TVariant...>>;
using FabDoubleScalarMap = FabNamedMap<double, std::vector<double>>;

template <typename... TVariant>
using FabVariantVector = std::vector<FabVariant<TVariant...>>;

// Highest accuracy without harming matrix inverse 1e-7
static constexpr auto FAB_EPS = 1e-6;

struct FabError : public std::runtime_error {
  explicit FabError(const std::string& error_msg) : std::runtime_error(error_msg) {}

  explicit FabError(const char* error_msg) : std::runtime_error(error_msg) {}

  static FabError customized(std::string expression, std::string message) {
    FabError error("");
    error.expression_ = std::move(expression);
    error.message_ = std::move(message);
    return error;
  }

  const char* what() const _NOEXCEPT override {
    static std::string full_message;
    full_message = expression_ + ": " + message_;
    return full_message.c_str();
  }

  std::string expression_;
  std::string message_;
};

struct FabParamNotFoundError : public std::runtime_error {
  explicit FabParamNotFoundError(const std::string& error_msg)
      : std::runtime_error(std::string("[Param not found]: ") + error_msg) {}

  explicit FabParamNotFoundError(const char* error_msg)
      : std::runtime_error(std::string("[Param not found]: ") + error_msg) {}
};
