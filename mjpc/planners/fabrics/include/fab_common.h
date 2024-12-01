#pragma once
// Dynamic Optimization Fabrics for Motion Generation
// https://arxiv.org/abs/2205.08454
// https://github.com/tud-amr/fabrics

#include <any>
#include <shared_mutex>
#include <variant>

#include "mjpc/casadi/casadi_common.h"

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
#define FAB_DRAW_TRAJECTORY (0)
#define FAB_OBSTACLE_SIZE_SCALE (1)

#define FAB_RANDOM_DETERMINISTIC (0)

using FabSharedMutexLock = std::shared_lock<std::shared_mutex>;
using FabMutexLock = std::lock_guard<std::mutex>;

using FabLinkCollisionProps = std::map<std::string, std::vector<double> /*size or radius*/>;

enum class FabControlMode : uint8_t { VEL, ACC };

struct FabError : public mjpc::MjpcError {
  explicit FabError(std::string error_msg = "") : MjpcError(std::move(error_msg)) {
    message_ = "[Fabrics] " + message_;
  }

  explicit FabError(const char* error_msg = nullptr) : FabError(std::string(error_msg)) {}
};

struct FabParamNotFoundError : public mjpc::MjpcParamNotFoundError {
  explicit FabParamNotFoundError(const std::string& error_msg)
      : MjpcParamNotFoundError("[Fabrics]: " + error_msg) {}

  explicit FabParamNotFoundError(const char* error_msg) : FabParamNotFoundError(std::string(error_msg)) {}
};
