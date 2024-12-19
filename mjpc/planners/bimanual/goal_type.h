#pragma once

#include <string>

class GoalType {
public:
  enum class Type : u_int8_t { GESTURE, GOTO, PLAN, GRASP, RELEASE, ERROR_RECOVERY, INVALID };

  // For accessing enums directly from outside
  using enum Type;

  GoalType();

  GoalType(const Type type): t_(type) {
  };

  GoalType& operator=(const Type type) {
    t_ = type;
    return *this;
  }

  bool operator==(const Type type) const {
    return t_ == type;
  }

  // For convenient implicit cast between [GoalType] <-> [Type]
  operator Type() const { return t_; };

  std::string to_string() const {
    switch (t_) {
      case Type::GESTURE:
        return "gesture";
      case Type::GOTO:
        return "goto";
      case Type::PLAN:
        return "plan";
      case Type::GRASP:
        return "grasp";
      case Type::RELEASE:
        return "release";
      case Type::ERROR_RECOVERY:
        return "error_recovery";
      default:
        return "INVALID";
    }
  }

private:
  Type t_;
};
