#pragma once

#include <optional>
#include <string>
#include <vector>

#include "mjpc/urdf_parser/include/common.h"
#include "mjpc/urdf_parser/include/txml.h"

namespace urdf {
struct JointDynamics {
  double damping = 0;
  double friction = 0;

  void clear() {
    damping = 0;
    friction = 0;
  }

  static std::shared_ptr<JointDynamics> fromXml(TiXmlElement* xml);
};

struct JointLimits {
  double lower = 0;
  double upper = 0;
  double effort = 0;
  double velocity = 0;

  void clear() {
    lower = 0;
    upper = 0;
    effort = 0;
    velocity = 0;
  }

  static std::shared_ptr<JointLimits> fromXml(TiXmlElement* xml);
};

struct JointSafety {
  double upper_limit = 0;
  double lower_limit = 0;
  double k_position = 0;
  double k_velocity = 0;

  void clear() {
    upper_limit = 0;
    lower_limit = 0;
    k_position = 0;
    k_velocity = 0;
  }

  static std::shared_ptr<JointSafety> fromXml(TiXmlElement* xml);
};

struct JointCalibration {
  std::optional<double> rising;
  std::optional<double> falling;

  void clear() {
    rising.reset();
    falling.reset();
  }
  static std::shared_ptr<JointCalibration> fromXml(TiXmlElement* xml);
};

struct JointMimic {
  std::string joint_name;
  double offset = 0.;
  double multiplier = 0.;

  void clear() {
    joint_name = "";
    offset = 0.;
    multiplier = 0.;
  }

  static std::shared_ptr<JointMimic> fromXml(TiXmlElement* xml);
};

enum class JointType : uint8_t {
  UNKNOWN,
  REVOLUTE,  // rotation axis
  CONTINUOUS,
  PRISMATIC,  // translation axis
  FLOATING,
  PLANAR,  // plane normal axis
  FIXED
};

struct Joint {
  std::string name;
  JointType type = JointType::UNKNOWN;
  Vector3 axis;
  std::string child_link_name;
  std::string parent_link_name;
  Transform parent_to_joint_transform;

  std::optional<std::shared_ptr<JointDynamics>> dynamics;
  std::optional<std::shared_ptr<JointLimits>> limits;
  std::optional<std::shared_ptr<JointSafety>> safety;
  std::optional<std::shared_ptr<JointCalibration>> calibration;
  std::optional<std::shared_ptr<JointMimic>> mimic;

  void clear() {
    this->axis.clear();
    this->child_link_name.clear();
    this->parent_link_name.clear();
    this->parent_to_joint_transform.clear();

    this->dynamics.reset();
    this->limits.reset();
    this->safety.reset();
    this->calibration.reset();
    this->type = JointType::UNKNOWN;
  }

  static std::shared_ptr<Joint> fromXml(TiXmlElement* xml);
};

using JointPtr = std::shared_ptr<Joint>;
}  // namespace urdf
