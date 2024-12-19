#pragma once

#include <string>

class ControllerType {
public:
  enum class Type : u_int8_t {
    RELATIVE_POSE,
    WHOLE_BODY_AVOIDANCE,
    ABSOLUTE_POSITION,
    EE_TILT,
    ROTATIONAL_ADMITTANCE,
    JOINT_LIMIT_AVOIDANCE,
    INVALID
  };

  // For accessing enums directly from outside
  using enum Type;

  explicit ControllerType(const Type type): t_(type) {
  };

  ControllerType& operator=(const Type type) {
    t_ = type;
    return *this;
  }

  bool operator==(const Type type) const {
    return t_ == type;
  }

  // For convenient implicit cast between [ControllerType] <-> [Type]
  operator Type() const { return t_; };

  std::string to_string() const {
    switch (t_) {
      case Type::RELATIVE_POSE:
        return "relative_pose";
      case Type::WHOLE_BODY_AVOIDANCE:
        return "whole_body_avoidance";
      case Type::ABSOLUTE_POSITION:
        return "absolute_position";
      case Type::EE_TILT:
        return "ee_tilt";
      case Type::ROTATIONAL_ADMITTANCE:
        return "rotational_admittance";
      case Type::JOINT_LIMIT_AVOIDANCE:
        return "joint_limit_avoidance";
      default:
        return "INVALID";
    }
  }

private:
  Type t_;
};
