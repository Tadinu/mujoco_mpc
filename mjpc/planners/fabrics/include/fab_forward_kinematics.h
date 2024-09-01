#pragma once

// casadi
#include <casadi/casadi.hpp>

// urdf_parser
#include "mjpc/urdf_parser/include/model.h"

// fabrics
#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_math_util.h"

enum class FabRobotBaseType : uint8_t { DIFF_DRIVE, HOLONOMIC };

class FabForwardKinematics {
public:
  virtual CaSX casadi(const CaSX& q, const FabVariant<int, std::string>& child_link,
                      const FabVariant<int, std::string>& parent_link = {},
                      const CaSX& link_transf = fab_math::CASX_TRANSF_IDENTITY, bool position_only = false) {
    return {};
  }

  virtual int n() { return n_; }

protected:
  int n_ = 0;
  CaSX mount_transformation_ = fab_math::CASX_TRANSF_IDENTITY;
};

using FabForwardKinematicsPtr = std::shared_ptr<FabForwardKinematics>;

class FabURDFForwardKinematics : public FabForwardKinematics {
public:
  FabURDFForwardKinematics() = default;

  FabURDFForwardKinematics(std::string urdf_file, std::string base_link_name,
                           std::vector<std::string> endtip_names,
                           const FabRobotBaseType base_type = FabRobotBaseType::HOLONOMIC)
      : urdf_file_(std::move(urdf_file)),
        base_link_name_(std::move(base_link_name)),
        endtip_names_(std::move(endtip_names)),
        base_type_(base_type) {
    assert(read_urdf());
    n_ = urdf_model_.get_dof();
    q_ca_ = CaSX::sym("q", n_);
    if (FabRobotBaseType::DIFF_DRIVE == base_type) {
      q_base_ = CaSX::sym("q_base", 3);
    }
    compose_functions();
  }

  int n() override { return (FabRobotBaseType::DIFF_DRIVE == base_type_) ? (n_ + 3) : n_; }

  urdf::UrdfModel urdf_model() const { return urdf_model_; }
  bool read_urdf() {
    urdf_model_.base_link_name = base_link_name_;
    urdf_model_.endtip_names = endtip_names_;
    // TODO: TAKE FROM PARAM
    urdf_model_.actuated_joint_types = {urdf::JointType::PRISMATIC, urdf::JointType::REVOLUTE,
                                        urdf::JointType::CONTINUOUS};
    return urdf_model_.fromUrdfFile(urdf_file_);
  }

  void compose_functions() {
    fks_.clear();
    for (const auto& link : urdf_model_.get_links()) {
      CaSX q;
      if (FabRobotBaseType::DIFF_DRIVE == base_type_) {
        q = CaSX::vertcat({q_base_, q_ca_});
      } else {
        q = q_ca_;
      }
      fks_[link->name] = CaFunction("fk" + link->name, {q}, {casadi(q, link->name)});
    }
  }

  // Returns the forward kinematics as a casadi function
  CaSX get_robot_fk(const std::string& base_name, const std::string& endtip_name, const CaSX& q,
                    const CaSX& link_transf = fab_math::CASX_TRANSF_IDENTITY) {
    const auto joint_list = urdf_model_.get_joints(urdf_model_.root_link->name, endtip_name);
    auto T_fk = fab_math::CASX_TRANSF_IDENTITY;
    for (const auto& joint : joint_list) {
      const auto& joint_transf = joint->parent_to_joint_transform;

      const auto& xyz = joint_transf.position;
      const auto& rpy = joint_transf.rotation.rpy;

      switch (joint->type) {
        case urdf::JointType::FIXED: {
          T_fk = CaSX::mtimes(T_fk, fab_math::transform(xyz, rpy));
        } break;

        case urdf::JointType::PRISMATIC: {
          const urdf::Vector3 axis =
              (joint->axis == urdf::Vector3::Zero) ? urdf::Vector3::UnitX : joint->axis;
          const auto joint_frame = fab_math::prismatic(
              xyz, rpy, axis, fab_core::get_casx(q, urdf_model_.joint_name_map[joint->name]));

#if 0
          FAB_PRINTDB("AXIS", axis.to_string());
          FAB_PRINTDB("JOINT FRAME", joint_frame, xyz.to_string(), rpy.to_string(),
                            urdf_model_.joint_name_map[joint->name],
                            fab_core::get_casx(q, urdf_model_.joint_name_map[joint->name]));
#endif
          T_fk = CaSX::mtimes(T_fk, joint_frame);
        } break;

        case urdf::JointType::REVOLUTE:
        case urdf::JointType::CONTINUOUS: {
          urdf::Vector3 axis = (joint->axis == urdf::Vector3::Zero) ? urdf::Vector3::UnitX : joint->axis;
          axis = double((1. / CaSX::norm_2(axis.to_vector())).scalar()) * axis;
          FAB_PRINT("get_robot_fk", joint->name, urdf_model_.joint_name_map[joint->name], xyz.to_string(),
                    rpy.to_string(), axis.to_string());
          const auto joint_frame = fab_math::revolute(
              xyz, rpy, axis, fab_core::get_casx(q, urdf_model_.joint_name_map[joint->name]));
          T_fk = CaSX::mtimes(T_fk, joint_frame);
        } break;

        default:
          break;
      }
    }
    return CaSX::mtimes(T_fk, link_transf);
  }

  CaSX casadi(const CaSX& q, const FabVariant<int, std::string>& child_link,
              const FabVariant<int, std::string>& parent_link = {},
              const CaSX& link_transf = fab_math::CASX_TRANSF_IDENTITY, bool position_only = false) override {
    auto parent_link_name = fab_core::get_variant_value<std::string>(parent_link);
    if (parent_link_name.empty()) {
      parent_link_name = base_link_name_;
    }

    const auto child_link_name = fab_core::get_variant_value<std::string>(child_link);
    if (!urdf_model_.get_link(child_link_name)) {
      throw FabError(child_link_name + " :Link not found in URDF model " + urdf_model_.name);
    }

    CaSX fk;
    switch (base_type_) {
      case FabRobotBaseType::DIFF_DRIVE: {
        fk = get_robot_fk(parent_link_name, child_link_name,
                          fab_core::get_casx(q, std::array<casadi_int, 2>{2, CASADI_INT_MAX}), link_transf);
        const CaSX q_2 = fab_core::get_casx(q, 2);
        const CaSX c = CaSX::cos(q_2);
        const CaSX s = CaSX::sin(q_2);
        const CaSX T_base = CaSX::blockcat({
            {c, -s, 0, fab_core::get_casx(q, 0)},
            {s, c, 0, fab_core::get_casx(q, 1)},
            {0, 0, 1, 0},
            {0, 0, 0, 1},
        });
        fk = CaSX::mtimes(T_base, fk);
      } break;
      default:
        fk = get_robot_fk(parent_link_name, child_link_name, q, link_transf);
        fk = CaSX::mtimes(mount_transformation_, fk);
        break;
    }

    if (position_only) {
      fk = fab_core::get_casx2(fk, {0, 3}, 3);
    }
    FAB_PRINT("URDFFK casadi", parent_link_name, child_link_name, q, fk);
    return fk;
  }

protected:
  urdf::UrdfModel urdf_model_;
  std::string urdf_file_;
  CaSX q_ca_;
  CaSX q_base_;
  std::string base_link_name_;
  std::vector<std::string> endtip_names_;
  FabRobotBaseType base_type_ = FabRobotBaseType::HOLONOMIC;
  std::map<std::string, CaFunction> fks_;
};
