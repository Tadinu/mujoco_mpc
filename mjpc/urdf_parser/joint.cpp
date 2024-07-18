#include "mjpc/urdf_parser/include/joint.h"

// abseil
#include "absl/strings/ascii.h"
#include "absl/types/any.h"
#include "absl/types/bad_any_cast.h"

namespace urdf {

const char *getParentJointName(TiXmlElement *xml) {
  // this should always be set since we check for the joint name in parseJoint already
  return ((TiXmlElement *)xml->Parent())->Attribute("name");
}

// ------------------- JointDynamics Implementation -------------------

std::shared_ptr<JointDynamics> JointDynamics::fromXml(TiXmlElement *xml) {
  std::shared_ptr<JointDynamics> jd = std::make_shared<JointDynamics>();
  const char *damping_str = xml->Attribute("damping");
  if (damping_str != nullptr) {
    try {
      jd->damping = absl::any_cast<double>(damping_str);
    } catch (const absl::bad_any_cast &e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "': dynamics damping value ("
                << damping_str << ") is not a float: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  const char *friction_str = xml->Attribute("friction");
  if (friction_str != nullptr) {
    try {
      jd->friction = absl::any_cast<double>(friction_str);
    } catch (const absl::bad_any_cast &e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "': dynamics friction value ("
                << friction_str << ") is not a float: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  if (damping_str == nullptr && friction_str == nullptr) {
    std::ostringstream error_msg;
    error_msg << "Error while parsing joint '" << getParentJointName(xml)
              << "': joint dynamics element specified with no damping and no friction!";
    throw URDFParseError(error_msg.str());
  }

  return jd;
}

// ------------------- JointLimits Implementation -------------------

std::shared_ptr<JointLimits> JointLimits::fromXml(TiXmlElement *xml) {
  std::shared_ptr<JointLimits> jl = std::make_shared<JointLimits>();

  const char *lower_str = xml->Attribute("lower");
  if (lower_str != nullptr) {
    try {
      jl->lower = absl::any_cast<double>(lower_str);
    } catch (const absl::bad_any_cast &e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "': limits lower value ("
                << lower_str << ") is not a float: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  const char *upper_str = xml->Attribute("upper");
  if (upper_str != nullptr) {
    try {
      jl->upper = absl::any_cast<double>(upper_str);
    } catch (const absl::bad_any_cast &e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "': limits upper value ("
                << upper_str << ") is not a float: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  const char *effort_str = xml->Attribute("effort");
  if (effort_str != nullptr) {
    try {
      jl->effort = absl::any_cast<double>(effort_str);
    } catch (const absl::bad_any_cast &e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' limits effort value ("
                << effort_str << ") is not a float: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing joint '" << getParentJointName(xml)
              << "' joint limit: no effort specified!";
    throw URDFParseError(error_msg.str());
  }

  const char *velocity_str = xml->Attribute("velocity");
  if (velocity_str != nullptr) {
    try {
      jl->velocity = absl::any_cast<double>(velocity_str);
    } catch (const absl::bad_any_cast &e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' limits velocity value ("
                << velocity_str << ") is not a float: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing joint '" << getParentJointName(xml)
              << "' joint limit: no velocity specified!";
    throw URDFParseError(error_msg.str());
  }

  return jl;
}

// ------------------- JointSafety Implementation -------------------

std::shared_ptr<JointSafety> JointSafety::fromXml(TiXmlElement *xml) {
  std::shared_ptr<JointSafety> js = std::make_shared<JointSafety>();

  const char *lower_limit_str = xml->Attribute("lower_limit");
  if (lower_limit_str != nullptr) {
    try {
      js->lower_limit = absl::any_cast<double>(lower_limit_str);
    } catch (const absl::bad_any_cast &e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' safety lower_limit value ("
                << lower_limit_str << ") is not a float: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  const char *upper_limit_str = xml->Attribute("upper_limit");
  if (upper_limit_str != nullptr) {
    try {
      js->upper_limit = absl::any_cast<double>(upper_limit_str);
    } catch (const absl::bad_any_cast &e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' safety upper_limit value ("
                << upper_limit_str << ") is not a float: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  const char *k_position_str = xml->Attribute("k_position");
  if (k_position_str != nullptr) {
    try {
      js->k_position = absl::any_cast<double>(k_position_str);
    } catch (const absl::bad_any_cast &e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' safety k_position value ("
                << k_position_str << ") is not a float: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  const char *k_velocity_str = xml->Attribute("k_velocity");
  if (k_velocity_str != nullptr) {
    try {
      js->k_velocity = absl::any_cast<double>(k_velocity_str);
    } catch (const absl::bad_any_cast &e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' safety k_velocity value ("
                << k_velocity_str << ") is not a float: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' joint safety no k_velocity!";
    throw URDFParseError(error_msg.str());
  }

  return js;
}

// ------------------- JointCalibration Implementation -------------------

std::shared_ptr<JointCalibration> JointCalibration::fromXml(TiXmlElement *xml) {
  std::shared_ptr<JointCalibration> jc = std::make_shared<JointCalibration>();

  const char *rising_str = xml->Attribute("rising");
  if (rising_str != nullptr) {
    try {
      jc->rising = absl::any_cast<double>(rising_str);
    } catch (const absl::bad_any_cast &e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml)
                << "' calibration rising_position value (" << rising_str << ") is not a float: " << e.what()
                << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  const char *falling_str = xml->Attribute("falling");
  if (falling_str != nullptr) {
    try {
      jc->falling = absl::any_cast<double>(falling_str);
    } catch (const absl::bad_any_cast &e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml)
                << "' calibration falling_position value (" << falling_str << ") is not a float: " << e.what()
                << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  return jc;
}

// ------------------- JointMimic Implementation -------------------

std::shared_ptr<JointMimic> JointMimic::fromXml(TiXmlElement *xml) {
  std::shared_ptr<JointMimic> jm = std::make_shared<JointMimic>();

  const char *joint_name_str = xml->Attribute("joint");
  if (joint_name_str != nullptr) {
    jm->joint_name = joint_name_str;
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing joint '" << getParentJointName(xml)
              << "joint mimic: no mimic joint specified!";
    throw URDFParseError(error_msg.str());
  }

  const char *multiplier_str = xml->Attribute("multiplier");
  if (multiplier_str != nullptr) {
    try {
      jm->multiplier = absl::any_cast<double>(multiplier_str);
    } catch (const absl::bad_any_cast &e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' mimic multiplier value ("
                << multiplier_str << ") is not a float: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  const char *offset_str = xml->Attribute("offset");
  if (offset_str != nullptr) {
    try {
      jm->offset = absl::any_cast<double>(offset_str);
    } catch (const absl::bad_any_cast &e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing joint '" << getParentJointName(xml) << "' mimic offset value ("
                << offset_str << ") is not a float: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  return jm;
}

// ------------------- Joint Implementation -------------------

std::shared_ptr<Joint> Joint::fromXml(TiXmlElement *xml) {
  std::shared_ptr<Joint> joint = std::make_shared<Joint>();

  const char *name = xml->Attribute("name");
  if (name != nullptr) {
    joint->name = name;
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing model: unnamed joint found!";
    throw URDFParseError(error_msg.str());
  }

  TiXmlElement *origin_xml = xml->FirstChildElement("origin");
  if (origin_xml != nullptr) {
    try {
      joint->parent_to_joint_transform = Transform::fromXml(origin_xml);
    } catch (urdf::URDFParseError &e) {
      std::ostringstream error_msg;
      error_msg << "Error! Malformed parent origin element for joint '" << joint->name << "': " << e.what()
                << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  TiXmlElement *parent_xml = xml->FirstChildElement("parent");
  if (parent_xml != nullptr) {
    const char *pname = parent_xml->Attribute("link");
    if (pname != nullptr) {
      joint->parent_link_name = std::string(pname);
    }
    // if no parent link name specified. this might be the root node
  }

  TiXmlElement *child_xml = xml->FirstChildElement("child");
  if (child_xml) {
    const char *pname = child_xml->Attribute("link");
    if (pname != nullptr) {
      joint->child_link_name = std::string(pname);
    }
  }

  const char *type_char = xml->Attribute("type");
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
    TiXmlElement *axis_xml = xml->FirstChildElement("axis");
    if (axis_xml == nullptr) {
      joint->axis = Vector3(1.0, 0.0, 0.0);
    } else {
      if (axis_xml->Attribute("xyz")) {
        try {
          joint->axis = Vector3::fromVecStr(axis_xml->Attribute("xyz"));
        } catch (URDFParseError &e) {
          std::ostringstream error_msg;
          error_msg << "Error! Malformed axis element for joint [" << joint->name << "]: " << e.what();
          throw URDFParseError(error_msg.str());
        }
      }
    }
  }

  TiXmlElement *prop_xml = xml->FirstChildElement("dynamics");
  if (prop_xml != nullptr) {
    joint->dynamics = JointDynamics::fromXml(prop_xml);
  }

  TiXmlElement *limit_xml = xml->FirstChildElement("limit");
  if (limit_xml != nullptr) {
    joint->limits = JointLimits::fromXml(limit_xml);
  }

  TiXmlElement *safety_xml = xml->FirstChildElement("safety_controller");
  if (safety_xml != nullptr) {
    joint->safety = JointSafety::fromXml(safety_xml);
  }

  TiXmlElement *calibration_xml = xml->FirstChildElement("calibration");
  if (calibration_xml != nullptr) {
    joint->calibration = JointCalibration::fromXml(calibration_xml);
  }

  TiXmlElement *mimic_xml = xml->FirstChildElement("mimic");
  if (mimic_xml != nullptr) {
    joint->mimic = JointMimic::fromXml(mimic_xml);
  }

  return joint;
}
}  // namespace urdf
