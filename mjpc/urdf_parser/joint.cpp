#include "mjpc/urdf_parser/include/joint.h"

// abseil
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"

namespace urdf {
const char* getParentJointName(TiXmlElement* xml) {
  // this should always be set since we check for the joint name in parseJoint already
  return ((TiXmlElement*)xml->Parent())->Attribute("name");
}

// ------------------- JointDynamics Implementation -------------------

std::shared_ptr<JointDynamics> JointDynamics::fromXml(TiXmlElement* xml) {
  std::shared_ptr<JointDynamics> jd = std::make_shared<JointDynamics>();
#define VERIFY_JOINT_DYNAMICS(attr, out)                                                  \
  {                                                                                       \
    const auto val_str = urdf::get_xml_attr(xml, attr);                                   \
    if (!val_str.empty()) {                                                               \
      if (!absl::SimpleAtod(val_str, &out)) {                                             \
        std::ostringstream error_msg;                                                     \
        error_msg << "Error while parsing joint dynamics'" << getParentJointName(xml)     \
                  << "': " + std::string(attr) + " (" << val_str << ") is not a double!"; \
        throw URDFParseError(error_msg.str());                                            \
      }                                                                                   \
    }                                                                                     \
  }

  VERIFY_JOINT_DYNAMICS("damping", jd->damping);
  VERIFY_JOINT_DYNAMICS("friction", jd->friction);

  return jd;
}

// ------------------- JointLimits Implementation -------------------

std::shared_ptr<JointLimits> JointLimits::fromXml(TiXmlElement* xml) {
  std::shared_ptr<JointLimits> jl = std::make_shared<JointLimits>();
#define VERIFY_JOINT_LIMITS(attr, out)                                                   \
  {                                                                                      \
    const auto val_str = urdf::get_xml_attr(xml, attr);                                  \
    if (!val_str.empty()) {                                                              \
      if (!absl::SimpleAtod(val_str, &out)) {                                            \
        std::ostringstream error_msg;                                                    \
        error_msg << "Error while parsing joint limits '" << getParentJointName(xml)     \
                  << "': " + std::string(attr) + " (" << val_str << ") is not a double"; \
        throw URDFParseError(error_msg.str());                                           \
      }                                                                                  \
    }                                                                                    \
  }
  VERIFY_JOINT_LIMITS("lower", jl->lower);
  VERIFY_JOINT_LIMITS("upper", jl->upper);
  VERIFY_JOINT_LIMITS("effort", jl->effort);
  VERIFY_JOINT_LIMITS("velocity", jl->velocity);

  return jl;
}

// ------------------- JointSafety Implementation -------------------

std::shared_ptr<JointSafety> JointSafety::fromXml(TiXmlElement* xml) {
  std::shared_ptr<JointSafety> js = std::make_shared<JointSafety>();

  const auto lower_limit = urdf::get_xml_attr(xml, "lower_limit");
  if (!lower_limit.empty()) {
    if (!absl::SimpleAtod(lower_limit, &js->lower_limit)) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' safety lower_limit value ("
                << lower_limit << ") is not a double";
      throw URDFParseError(error_msg.str());
    }
  }

  const auto upper_limit = urdf::get_xml_attr(xml, "upper_limit");
  if (!upper_limit.empty()) {
    if (!absl::SimpleAtod(upper_limit, &js->upper_limit)) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' safety upper_limit value ("
                << upper_limit << ") is not a double";
      throw URDFParseError(error_msg.str());
    }
  }

  const auto k_position = urdf::get_xml_attr(xml, "k_position");
  if (!k_position.empty()) {
    if (!absl::SimpleAtod(k_position, &js->k_position)) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' safety k_position value ("
                << k_position << ") is not a double";
      throw URDFParseError(error_msg.str());
    }
  }

  const auto k_velocity = urdf::get_xml_attr(xml, "k_velocity");
  if (!k_velocity.empty()) {
    if (!absl::SimpleAtod(k_velocity, &js->k_velocity)) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' safety k_velocity value ("
                << k_velocity << ") is not a double";
      throw URDFParseError(error_msg.str());
    }
  }
  return js;
}

// ------------------- JointCalibration Implementation -------------------

std::shared_ptr<JointCalibration> JointCalibration::fromXml(TiXmlElement* xml) {
  std::shared_ptr<JointCalibration> jc = std::make_shared<JointCalibration>();

  const auto rising = urdf::get_xml_attr(xml, "rising");
  if (!rising.empty()) {
    if (!absl::SimpleAtod(rising, &jc->rising.value())) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml)
                << "' calibration rising_position value (" << rising << ") is not a double";
      throw URDFParseError(error_msg.str());
    }
  }

  const auto falling = urdf::get_xml_attr(xml, "falling");
  if (!falling.empty()) {
    if (!absl::SimpleAtod(falling, &jc->falling.value())) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml)
                << "' calibration falling_position value (" << falling << ") is not a double";
      throw URDFParseError(error_msg.str());
    }
  }

  return jc;
}

// ------------------- JointMimic Implementation -------------------

std::shared_ptr<JointMimic> JointMimic::fromXml(TiXmlElement* xml) {
  std::shared_ptr<JointMimic> jm = std::make_shared<JointMimic>();

  const char* joint_name_str = xml->Attribute("joint");
  if (joint_name_str != nullptr) {
    jm->joint_name = joint_name_str;
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing joint '" << getParentJointName(xml)
              << "joint mimic: no mimic joint specified!";
    throw URDFParseError(error_msg.str());
  }

  const auto multiplier = urdf::get_xml_attr(xml, "multiplier");
  if (!multiplier.empty()) {
    if (!absl::SimpleAtod(multiplier, &jm->multiplier)) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' mimic multiplier value ("
                << multiplier << ") is not a double";
      throw URDFParseError(error_msg.str());
    }
  }

  const auto offset = urdf::get_xml_attr(xml, "offset");
  if (!offset.empty()) {
    if (!absl::SimpleAtod(offset, &jm->offset)) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' mimic offset value ("
                << offset << ") is not a double";
      throw URDFParseError(error_msg.str());
    }
  }

  return jm;
}

// ------------------- Joint Implementation -------------------

std::shared_ptr<Joint> Joint::fromXml(TiXmlElement* xml) {
  std::shared_ptr<Joint> joint = std::make_shared<Joint>();

  const char* name = xml->Attribute("name");
  if (name != nullptr) {
    joint->name = name;
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing model: unnamed joint found!";
    throw URDFParseError(error_msg.str());
  }

  TiXmlElement* origin_xml = xml->FirstChildElement("origin");
  if (origin_xml != nullptr) {
    try {
      joint->parent_to_joint_transform = Transform::fromXml(origin_xml);
    } catch (urdf::URDFParseError& e) {
      std::ostringstream error_msg;
      error_msg << "Error! Malformed parent origin element for joint '" << joint->name << "': " << e.what()
                << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  TiXmlElement* parent_xml = xml->FirstChildElement("parent");
  if (parent_xml != nullptr) {
    const char* pname = parent_xml->Attribute("link");
    if (pname != nullptr) {
      joint->parent_link_name = std::string(pname);
    }
    // if no parent link name specified. this might be the root node
  }

  TiXmlElement* child_xml = xml->FirstChildElement("child");
  if (child_xml) {
    const char* pname = child_xml->Attribute("link");
    if (pname != nullptr) {
      joint->child_link_name = std::string(pname);
    }
  }

  const char* type_char = xml->Attribute("type");
  if (type_char == nullptr) {
    std::ostringstream error_msg;
    error_msg << "Error! Joint " << joint->name << " has no type, check to see if it's a reference.";
    throw URDFParseError(error_msg.str());
  }

  std::string type_str = type_char;
  if (type_str == "planar")
    joint->type = JointType::PLANAR;
  else if (type_str == "floating")
    joint->type = JointType::FLOATING;
  else if (type_str == "revolute")
    joint->type = JointType::REVOLUTE;
  else if (type_str == "continuous")
    joint->type = JointType::CONTINUOUS;
  else if (type_str == "prismatic")
    joint->type = JointType::PRISMATIC;
  else if (type_str == "fixed")
    joint->type = JointType::FIXED;
  else {
    std::ostringstream error_msg;
    error_msg << "Error! Joint '" << joint->name << "' has unknown type (" << type_str << ")!";
    throw URDFParseError(error_msg.str());
  }

  if (joint->type != JointType::FLOATING && joint->type != JointType::FIXED) {
    TiXmlElement* axis_xml = xml->FirstChildElement("axis");
    if (axis_xml == nullptr) {
      joint->axis = Vector3(1.0, 0.0, 0.0);
    } else {
      if (axis_xml->Attribute("xyz")) {
        try {
          joint->axis = Vector3::fromVecStr(axis_xml->Attribute("xyz"));
        } catch (URDFParseError& e) {
          std::ostringstream error_msg;
          error_msg << "Error! Malformed axis element for joint [" << joint->name << "]: " << e.what();
          throw URDFParseError(error_msg.str());
        }
      }
    }
  }

  TiXmlElement* prop_xml = xml->FirstChildElement("dynamics");
  if (prop_xml != nullptr) {
    joint->dynamics = JointDynamics::fromXml(prop_xml);
  }

  TiXmlElement* limit_xml = xml->FirstChildElement("limit");
  if (limit_xml != nullptr) {
    joint->limits = JointLimits::fromXml(limit_xml);
  }

  TiXmlElement* safety_xml = xml->FirstChildElement("safety_controller");
  if (safety_xml != nullptr) {
    joint->safety = JointSafety::fromXml(safety_xml);
  }

  TiXmlElement* calibration_xml = xml->FirstChildElement("calibration");
  if (calibration_xml != nullptr) {
    joint->calibration = JointCalibration::fromXml(calibration_xml);
  }

  TiXmlElement* mimic_xml = xml->FirstChildElement("mimic");
  if (mimic_xml != nullptr) {
    joint->mimic = JointMimic::fromXml(mimic_xml);
  }

  return joint;
}
}  // namespace urdf
