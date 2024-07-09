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
                      const FabVariant<int, std::string>& parent_link = FabVariant<int, std::string>(),
                      CaSX link_transf = CaSX::eye(4), bool position_only = false) {
    return CaSX();
  }

  virtual int n() { return n_; }

 protected:
  int n_ = 0;
  CaSX mount_transformation_ = CaSX::eye(4);
};

class FabURDFForwardKinematics : public FabForwardKinematics {
 public:
  FabURDFForwardKinematics() = default;

  FabURDFForwardKinematics(const std::string& urdf_file, const std::string& root_link_name,
                           const std::vector<std::string> endtip_names,
                           const FabRobotBaseType base_type = FabRobotBaseType::HOLONOMIC)
      : urdf_file_(urdf_file),
        root_link_name_(root_link_name),
        endtip_names_(endtip_names),
        base_type_(base_type) {
    read_urdf();
    // n_ = robot.degrees_of_freedom();
    q_ca_ = CaSX::sym("q", n_);
    if (base_type == FabRobotBaseType::DIFF_DRIVE) {
      q_base_ = CaSX::sym("q_base", 3);
      generate_functions();
    }
  }

  int n() override { return (base_type_ == FabRobotBaseType::DIFF_DRIVE) ? (n_ + 3) : n_; }

  void read_urdf() { robot_model_.fromUrdfFile(urdf_file_); }

  void generate_functions() {
    fks_.clear();
    for (const auto& link : robot_model_.get_links()) {
      CaSX q;
      if (base_type_ == FabRobotBaseType::DIFF_DRIVE) {
        q = CaSX::vertcat({q_base_, q_ca_});
      } else {
        q = q_ca_;
      }
      fks_[link->name] = CaFunction(std::string("fk") + link->name, {q}, {casadi(q, link->name)});
    }
  }

  // Returns the forward kinematics as a casadi function
  CaSX get_robot_fk(const std::string& base_name, const std::string& endtip_name, const CaSX& q,
                    const CaSX& link_transf = CaSX::eye(4)) {
    const auto joint_list = robot_model_.get_joints(robot_model_.root_link->name, endtip_name);
    auto T_fk = CaSX::eye(4);
    for (const auto& joint : joint_list) {
      const auto& joint_transf = joint->parent_to_joint_transform;

      const auto& xyz = joint_transf.position;
      const auto& rpy = joint_transf.rotation.rpy;

      switch (joint->type) {
        case urdf::JointType::FIXED: {
          T_fk = CaSX::mtimes(T_fk, fab_math::transform(xyz, rpy));
        } break;

        case urdf::JointType::PRISMATIC: {
          const CaSX axis = (joint->axis == urdf::Vector3::Zero) ? CaSX(urdf::Vector3::UnitX.to_vector())
                                                                 : CaSX(joint->axis.to_vector());
          const auto joint_frame = fab_math::prismatic(
              xyz, rpy, joint->axis, fab_core::get_casx(q, robot_model_.joint_name_map[joint->name]));
          T_fk = CaSX::mtimes(T_fk, joint_frame);
        } break;

        case urdf::JointType::REVOLUTE:
        case urdf::JointType::CONTINUOUS: {
          CaSX axis = (joint->axis == urdf::Vector3::Zero) ? CaSX(urdf::Vector3::UnitX.to_vector())
                                                           : CaSX(joint->axis.to_vector());
          axis = (1. / CaSX::norm_1(axis)) * axis;
          const auto joint_frame = fab_math::revolute(
              xyz, rpy, joint->axis, fab_core::get_casx(q, robot_model_.joint_name_map[joint->name]));
          T_fk = CaSX::mtimes(T_fk, joint_frame);
        } break;

        default:
          break;
      }
    }

    return CaSX::mtimes(T_fk, link_transf);
  }

  CaSX casadi(const CaSX& q, const FabVariant<int, std::string>& child_link,
              const FabVariant<int, std::string>& parent_link = FabVariant<int, std::string>(),
              CaSX link_transf = CaSX::eye(4), bool position_only = false) override {
    std::string parent_link_name = std::get<std::string>(parent_link);
    if (parent_link_name.empty()) {
      parent_link_name = root_link_name_;
    }

    const std::string child_link_name = std::get<std::string>(child_link);
    if (!robot_model_.get_link(child_link_name)) {
      throw FabError(child_link_name + " :Link not found in URDF model");
    }

    CaSX fk;
    if (base_type_ == FabRobotBaseType::DIFF_DRIVE) {
      fk = get_robot_fk(parent_link_name, child_link_name,
                        fab_core::get_casx(q, 2, std::numeric_limits<casadi_int>::max()), link_transf);
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
    } else {
      fk = get_robot_fk(parent_link_name, child_link_name, q, link_transf);
      fk = CaSX::mtimes(mount_transformation_, fk);
    }
    return fk;
  }

 protected:
  urdf::UrdfModel robot_model_;
  std::string urdf_file_;
  CaSX q_ca_;
  CaSX q_base_;
  std::string root_link_name_;
  std::vector<std::string> endtip_names_;
  FabRobotBaseType base_type_;
  std::map<std::string, CaFunction> fks_;
};
