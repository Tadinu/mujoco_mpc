#pragma once

#include <any>
#include <map>
#include <shared_mutex>
#include <variant>
#include <vector>

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

#define MJPC_DEBUG (0)

namespace mjpc {
using MjpcSharedMutexLock = std::shared_lock<std::shared_mutex>;
using MjpcMutexLock = std::lock_guard<std::mutex>;

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
} // namespace mjpc
