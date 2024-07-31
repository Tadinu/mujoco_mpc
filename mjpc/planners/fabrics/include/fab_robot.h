#pragma once

#include <absl/strings/match.h>
#include <mujoco/mujoco.h>

#include <casadi/casadi.hpp>
#include <filesystem>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_config.h"
#include "mjpc/planners/fabrics/include/fab_energized_geometry.h"
#include "mjpc/planners/fabrics/include/fab_energy.h"
#include "mjpc/planners/fabrics/include/fab_forward_kinematics.h"
#include "mjpc/planners/fabrics/include/fab_geometry.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"

class FabRobot {
public:
  FabRobot() = default;
  FabRobot(int dof, std::string model_path, std::string base_link_name, std::vector<std::string> endtip_names)
      : dof_(dof), model_path_(std::move(model_path)) {
    // 1- FK
    if (absl::EndsWith(model_path_, ".urdf")) {
      fk_ = std::make_shared<FabURDFForwardKinematics>(model_path_, std::move(base_link_name),
                                                       std::move(endtip_names));
      FAB_PRINT(model_name() + ": full-dof " + std::to_string(dof_) + " vs " +
                "actuated active dof: " + std::to_string(fk_->n()));
    }

    // 2- Casadi vars
    vars_ = std::make_shared<FabVariables>(
        CaSXDict{{"q", CaSX::sym("q", dof_)}, {"qdot", CaSX::sym("qdot", dof_)}});
    vars_->print_self();

    // 3- Base weighted geometry
    init_base_geometry();
  }

  void init_base_geometry() {
    const auto q = vars_->position_var();
    const auto qdot = vars_->velocity_var();
    // const auto new_parameters, base_energy =  parse_symbolic_input(self._config.base_energy, q, qdot);
    // vars_->add_parameters(new_parameters);
    auto base_geometry =
        std::make_shared<FabGeometry>(FabGeometryArgs{{"h", CaSX::zeros(dof_)}, {"var", vars_}});
    auto base_lagrangian =
        std::make_shared<FabLagrangian>(config_.base_energy(qdot), FabLagrangianArgs{{"var", vars_}});
    geometry_ = std::make_shared<FabWeightedSpec>(
        FabWeightedSpecArgs{{"g", std::move(base_geometry)}, {"le", std::move(base_lagrangian)}});
  }

  std::string model_name() const { return std::filesystem::path(model_path_).stem().string(); }
  int dof() const { return dof_; }
  FabVariablesPtr vars() const { return vars_; }
  FabForwardKinematicsPtr fk() const { return fk_; }
  FabWeightedSpecPtr weighted_geometry() const { return geometry_; }

protected:
  std::string model_path_;
  int dof_ = 0;
  FabVariablesPtr vars_ = nullptr;
  FabForwardKinematicsPtr fk_ = nullptr;
  FabWeightedSpecPtr geometry_ = nullptr;
  FabPlannerConfig config_;
};

using FabRobotPtr = std::shared_ptr<FabRobot>;
