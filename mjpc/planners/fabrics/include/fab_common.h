#pragma once
// Dynamic Optimization Fabrics for Motion Generation
// https://arxiv.org/abs/2205.08454
// https://github.com/tud-amr/fabrics

#include <any>
#include <casadi/casadi.hpp>
#include <shared_mutex>
#include <variant>

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
static constexpr auto CASADI_INT_MIN = std::numeric_limits<casadi_int>::min();
static constexpr auto CASADI_INT_MAX = std::numeric_limits<casadi_int>::max();

template <typename... TVariant>
using FabVariant = std::variant<std::monostate, TVariant...>;

template <typename... TVariant>
using FabNamedVariantPair = std::pair<std::string, FabVariant<TVariant...>>;
template <typename... TVariant>
using FabNamedMap = std::map<std::string, FabVariant<TVariant...>>;
using FabDoubleScalarMap = FabNamedMap<double, std::vector<double>>;
using FabNamedAnyMap = std::map<std::string, std::any>;

template <typename... TVariant>
using FabVariantVector = std::vector<FabVariant<TVariant...>>;

// Highest accuracy without harming matrix inverse 1e-7
static constexpr auto FAB_EPS = 1e-6;

#define FAB_DEBUG (0)
#define FAB_VERIFY_TUNED_PARAMS (1)
#define FAB_USE_ACTUATOR_VELOCITY (1)
#define FAB_USE_ACTUATOR_MOTOR (!FAB_USE_ACTUATOR_VELOCITY)
#define FAB_DRAW_TRAJECTORY (1)
#define FAB_OBSTACLE_SIZE_SCALE (1)

// NOTE: Dynamic goal is not yet working
#define FAB_DYNAMIC_GOAL_SUPPORTED (0)

#define FAB_RANDOM_DETERMINISTIC (0)

using FabSharedMutexLock = std::shared_lock<std::shared_mutex>;
using FabMutexLock = std::lock_guard<std::mutex>;

using FabLinkCollisionProps = std::map<std::string, std::vector<double> /*size or radius*/>;

struct FabError : public std::runtime_error {
  explicit FabError(const std::string& error_msg) : std::runtime_error(error_msg), message_(error_msg) {}

  explicit FabError(const char* error_msg) : std::runtime_error(error_msg), message_(error_msg) {}

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
      : std::runtime_error("[Param not found]: " + error_msg) {}

  explicit FabParamNotFoundError(const char* error_msg)
      : std::runtime_error("[Param not found]: " + std::string(error_msg)) {}
};
