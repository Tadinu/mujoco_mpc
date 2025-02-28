#pragma once

#include <any>
#include <map>
#include <shared_mutex>
#include <variant>
#include <vector>
#include <functional>
#include <memory>

// MuJoCo
#include <mujoco/mujoco.h>
#define MJPC_DEBUG (0)

namespace mjpc {
template <typename... TVariant>
using MjpcVariant = std::variant<std::monostate, TVariant...>;
template <typename... TVariant>
using MjpcVariantVector = std::vector<MjpcVariant<TVariant...>>;

template <typename... TVariant>
using MjpcNamedVariantPair = std::pair<std::string, MjpcVariant<TVariant...>>;
template <typename... TVariant>
using MjpcNamedMap = std::map<std::string, MjpcVariant<TVariant...>>;
using MjpcDoubleScalarMap = MjpcNamedMap<double, std::vector<double>>;
using MjpcNamedAnyMap = std::map<std::string, std::any>;

using MjpcSharedMutexLock = std::shared_lock<std::shared_mutex>;
using MjpcMutexLock = std::lock_guard<std::mutex>;

static constexpr const mjtNum* UNIT_X = (mjtNum[]){1, 0, 0};
static constexpr const mjtNum* UNIT_Y = (mjtNum[]){0, 1, 0};
static constexpr const mjtNum* UNIT_Z = (mjtNum[]){0, 0, 1};

enum class MjOwnerAppType : int8_t {
  MJAPP,
  MJPC
};

struct MjpcError : public std::runtime_error {
  explicit MjpcError(std::string error_msg = "")
    : std::runtime_error(error_msg), message_(std::move(error_msg)) {
  }

  explicit MjpcError(const char* error_msg = nullptr) : MjpcError(std::string(error_msg)) {
  }

  static MjpcError customized(std::string expression, std::string message) {
    MjpcError error(std::move(message));
    error.expression_ = std::move(expression);
    return error;
  }

#if 0  // clang
  const char* what() const _NOEXCEPT override {
#else
  const char* what() const _GLIBCXX_TXN_SAFE_DYN _GLIBCXX_NOTHROW override {
#endif
    static std::string full_message;
    full_message = expression_ + ": " + message_;
    return full_message.c_str();
  }

protected:
  std::string expression_;
  std::string message_;
};

struct MjpcParamNotFoundError : public std::runtime_error {
  explicit MjpcParamNotFoundError(const std::string& error_msg)
    : std::runtime_error("[Param not found]: " + error_msg) {
  }

  explicit MjpcParamNotFoundError(const char* error_msg)
    : std::runtime_error("[Param not found]: " + std::string(error_msg)) {
  }
};

class BaseSolver {
public:
  BaseSolver() = default;
  virtual ~BaseSolver() = default;
};

using BaseSolverPtr = std::shared_ptr<BaseSolver>;
using MjpcPlannerControlCb = std::function<std::vector<double>(double* policy_action, mjData* data,
                                                               const BaseSolverPtr& solver)>;
} // namespace mjpc
