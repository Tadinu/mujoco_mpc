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
  virtual bool init() = 0;
  virtual bool read_entity_model() = 0;
  virtual CaSX casadi(const CaSX& q, const FabVariant<int, std::string>& child_link,
                      const FabVariant<int, std::string>& parent_link = {},
                      const CaSX& link_transf = fab_math::CASX_TRANSF_IDENTITY,
                      const CaSX& link_transf_offset = fab_math::CASX_TRANSF_IDENTITY,
                      bool position_only = false) = 0;

  virtual int n() { return n_; }

protected:
  int n_ = 0;
  CaSX mount_transformation_ = fab_math::CASX_TRANSF_IDENTITY;
};

using FabForwardKinematicsPtr = std::shared_ptr<FabForwardKinematics>;

class FabURDFForwardKinematics : public FabForwardKinematics {
public:
  FabURDFForwardKinematics() = default;

  FabURDFForwardKinematics(std::string entity_model_file, std::string base_link_name,
                           std::vector<std::string> endtip_names,
                           const FabRobotBaseType base_type = FabRobotBaseType::HOLONOMIC)
    : entity_model_file_(std::move(entity_model_file)),
      base_link_name_(std::move(base_link_name)),
      endtip_names_(std::move(endtip_names)),
      base_type_(base_type) {
  }

  bool init() override {
    // 1.1- Read entity model from description file (urdf, xml, etc.)
    if (entity_model_file_.empty()) {
      MJPC_PRINT("[FabURDFForwardKinematics] entity model path is empty, failed reading MODEL");
      return false;
    }
    if (!read_entity_model()) {
      return false;
    }

    // 1.2- Init properties
    n_ = entity_model_->GetDof();
    q_ca_ = CaSX::sym("q", n_);

    if (FabRobotBaseType::DIFF_DRIVE == base_type_) {
      q_base_ = CaSX::sym("q_base", 3);
    }

    // 2- Compose symbolic casadi functions
    compose_functions();
    return true;
  }

  bool read_entity_model() override {
    if (false == read_urdf()) {
      MJPC_PRINT("[FabURDFForwardKinematics] failed reading MODEL", model_path());
      return false;
    }
    return true;
  }

  int n() override { return (FabRobotBaseType::DIFF_DRIVE == base_type_) ? (n_ + 3) : n_; }

  urdf::UrdfModel urdf_model() const { return *entity_model_; }
  std::string model_path() const { return entity_model_file_; }

  bool read_urdf() {
    entity_model_ = std::make_shared<urdf::UrdfModel>();
    entity_model_->base_link_name = base_link_name_;
    entity_model_->endtip_names = endtip_names_;
    // TODO: TAKE FROM PARAM
    entity_model_->actuated_joint_types = {urdf::JointType::PRISMATIC, urdf::JointType::REVOLUTE,
                                           urdf::JointType::CONTINUOUS};
    return entity_model_->FromUrdfFile(entity_model_file_);
  }

  void compose_functions() {
    fks_.clear();
    for (const auto& link : entity_model_->GetLinks()) {
      CaSX q;
      if (FabRobotBaseType::DIFF_DRIVE == base_type_) {
        q = CaSX::vertcat({q_base_, q_ca_});
      } else {
        q = q_ca_;
      }
      fks_[link->name] = CaFunction("fk" + link->name, {q}, {casadi(q, link->name)});
    }
  }

  // Returns the forward kinematics as a casadi function of [q]
  CaSX get_robot_fk(const std::string& base_name, const std::string& endtip_name, const CaSX& q,
                    const CaSX& link_transf = fab_math::CASX_TRANSF_IDENTITY,
                    const CaSX& link_transf_offset = fab_math::CASX_TRANSF_IDENTITY) {
    // NOTE: [base_name] is not always [entity_model_->root_link->name]
    const auto joint_list = entity_model_->GetJoints(base_name, endtip_name);
    auto T_fk = fab_math::CASX_TRANSF_IDENTITY;
    for (const auto& joint : joint_list) {
      const auto& joint_transf = joint->parent_to_joint_transform;

      const auto& xyz = joint_transf.position;
      const auto& rpy = joint_transf.rotation.rpy;

      switch (joint->type) {
        case urdf::JointType::FIXED: {
          T_fk = CaSX::mtimes(T_fk, fab_math::transform(xyz, rpy));
        }
        break;

        case urdf::JointType::PRISMATIC: {
          const urdf::Vector3 axis =
              (joint->axis == urdf::Vector3::Zero) ? urdf::Vector3::UnitX : joint->axis;
          const auto joint_frame = fab_math::prismatic(
              xyz, rpy, axis, mjpc::get_casx(q, entity_model_->joint_name_map[joint->name]));

#if 0
          MJPC_PRINTDB("AXIS", axis.to_string());
          MJPC_PRINTDB("JOINT FRAME", joint_frame, xyz.to_string(), rpy.to_string(),
                            entity_model_->joint_name_map[joint->name],
                            mjpc::get_casx(q, entity_model_->joint_name_map[joint->name]));
#endif
          T_fk = CaSX::mtimes(T_fk, joint_frame);
        }
        break;

        case urdf::JointType::REVOLUTE:
        case urdf::JointType::CONTINUOUS: {
          urdf::Vector3 axis = (joint->axis == urdf::Vector3::Zero) ? urdf::Vector3::UnitX : joint->axis;
          axis = double((1. / CaSX::norm_2(axis.to_vector())).scalar()) * axis;
          MJPC_PRINTDB("get_robot_fk", joint->name, joint->joint_type_name(),
                       entity_model_->joint_name_map[joint->name], xyz.to_string(), rpy.to_string(),
                       axis.to_string());
          const auto joint_frame = fab_math::revolute(
              xyz, rpy, axis, mjpc::get_casx(q, entity_model_->joint_name_map[joint->name]));
          T_fk = CaSX::mtimes(T_fk, joint_frame);
        }
        break;

        default:
          break;
      }
    }
    return CaSX::mtimes(CaSX::mtimes(T_fk, link_transf), link_transf_offset);
  }

  CaSX casadi(const CaSX& q, const FabVariant<int, std::string>& child_link,
              const FabVariant<int, std::string>& parent_link = {},
              const CaSX& link_transf = fab_math::CASX_TRANSF_IDENTITY,
              const CaSX& link_transf_offset = fab_math::CASX_TRANSF_IDENTITY,
              bool position_only = false) override {
    CaSX fk = position_only ? mjpc::CASX_3D_ZERO : fab_math::CASX_TRANSF_IDENTITY;

    auto parent_link_name = mjpc::get_variant_value<std::string>(parent_link);
    if (parent_link_name.empty()) {
      parent_link_name = base_link_name_;
    }

    const auto child_link_name = mjpc::get_variant_value<std::string>(child_link);
    if (!entity_model_->GetLink(child_link_name)) {
      throw FabError(child_link_name + " :Link not found in robot model " + model_path());
    } else if ((child_link_name == entity_model_->root_link->name) || (child_link_name == "world")) {
      return fk;
    }

    switch (base_type_) {
      case FabRobotBaseType::DIFF_DRIVE: {
        fk = get_robot_fk(parent_link_name, child_link_name,
                          mjpc::get_casx(q, std::array<casadi_int, 2>{2, CASADI_INT_MAX}), link_transf);
        const CaSX q_2 = mjpc::get_casx(q, 2);
        const CaSX c = CaSX::cos(q_2);
        const CaSX s = CaSX::sin(q_2);
        const CaSX T_base = CaSX::blockcat({
            {c, -s, 0, mjpc::get_casx(q, 0)},
            {s, c, 0, mjpc::get_casx(q, 1)},
            {0, 0, 1, 0},
            {0, 0, 0, 1},
        });
        fk = CaSX::mtimes(T_base, fk);
      }
      break;
      default:
        fk = get_robot_fk(parent_link_name, child_link_name, q, link_transf);
        fk = CaSX::mtimes(mount_transformation_, fk);
        break;
    }

    // Offset
    if (position_only) {
      fk = mjpc::get_casx2(fk, {0, 3}, 3) + (link_transf_offset.is_zero()
                                               ? mjpc::CASX_3D_ZERO
                                               : mjpc::get_casx2(link_transf_offset, {0, 3}, 3));
    } else {
      fk = CaSX::mtimes(fk, link_transf_offset);
    }
    MJPC_PRINTDB("URDFFK casadi", parent_link_name, child_link_name, q, fk);
    MJPC_PRINTDB("FK Offset:", link_transf_offset);
    return fk;
  }

protected:
  urdf::UrdfModelPtr entity_model_ = nullptr;
  std::string entity_model_file_;
  CaSX q_ca_;
  CaSX q_base_;
  std::string base_link_name_;
  std::vector<std::string> endtip_names_;
  FabRobotBaseType base_type_ = FabRobotBaseType::HOLONOMIC;
  std::map<std::string, CaFunction> fks_;
};
