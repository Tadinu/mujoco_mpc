#pragma once

// casadi
#include <casadi/casadi.hpp>

// mujoco
#include <mujoco/mujoco.h>

// fabrics
#include "mjpc/mjcf/mjcf_model.h"
#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_forward_kinematics.h"
#include "mjpc/planners/fabrics/include/fab_math_util.h"
#include "mjpc/utilities.h"

class FabMJCFForwardKinematics : public FabURDFForwardKinematics {
public:
  FabMJCFForwardKinematics() = default;

  FabMJCFForwardKinematics(std::string entity_model_file, std::string base_link_name,
                           std::vector<std::string> endtip_names,
                           const FabRobotBaseType base_type = FabRobotBaseType::HOLONOMIC)
      : FabURDFForwardKinematics(std::move(entity_model_file), std::move(base_link_name),
                                 std::move(endtip_names), base_type) {}

  bool read_entity_model() override {
    if (false == read_mjcf()) {
      MJPC_PRINT("[FabMJCFForwardKinematics] failed reading MODEL", model_path());
      return false;
    }
    return true;
  }

  mjpc::MjcfModel mjcf_model() const { return *mjcf_model_; }
  bool read_mjcf() {
    entity_model_ = std::make_shared<mjpc::MjcfModel>();
    mjcf_model_ = std::dynamic_pointer_cast<mjpc::MjcfModel>(entity_model_);
    mjcf_model_->base_link_name = base_link_name_;
    mjcf_model_->endtip_names = endtip_names_;
    return mjcf_model_->fromMjcfFile(entity_model_file_);
  }

protected:
  mjpc::MjcfModelPtr mjcf_model_ = nullptr;
};
