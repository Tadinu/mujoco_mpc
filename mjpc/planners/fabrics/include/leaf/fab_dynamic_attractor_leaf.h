#pragma once

#include <casadi/casadi.hpp>
#include <map>
#include <memory>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_config.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_energy.h"
#include "mjpc/planners/fabrics/include/fab_parameterized_map.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"
#include "mjpc/planners/fabrics/include/leaf/fab_dynamic_leaf.h"

/* FabGenericDynamicAttractorLeaf is a leaf to the tree of fabrics.
 * The attractor's potential and metric are defined through the corresponding symbolic expression functions
 */
class FabGenericDynamicAttractorLeaf : public FabDynamicLeaf {
public:
  FabGenericDynamicAttractorLeaf() = default;

  FabGenericDynamicAttractorLeaf(FabVariablesPtr root_variables, const CaSX& fk_goal,
                                 const std::string& attractor_name)
      : FabDynamicLeaf(std::move(root_variables), attractor_name + "_leaf", fk_goal.size().first,
                       fk_goal.size().first, fk_goal) {
    set_forward_map(attractor_name);
  }

  void set_potential(const FabConfigFunc& potential) override {
    const auto [x_potential, var_names] = potential(x_rel_, xdot_rel_, leaf_name_);
    parent_vars_->add_parameters(fab_core::parse_symbolic_casx(x_potential, var_names));
    const CaSX psi = weight_var_ * x_potential;
    CaSX h_psi = CaSX::gradient(psi, x_rel_);
    geom_ = std::make_shared<FabGeometry>(FabGeometryArgs{{"h", std::move(h_psi)}, {"var", relative_vars_}});
  }

  void set_metric(const FabConfigFunc& metric) override {
    const auto [attractor_metric, var_names] = metric(x_rel_, xdot_rel_, leaf_name_);
    // TODO: Check if needed to add params to [parent_vars_]
    parent_vars_->add_parameters(fab_core::parse_symbolic_casx(attractor_metric, var_names));
    const auto lagrangian_psi = CaSX::dot(xdot_rel_, CaSX::mtimes(attractor_metric, xdot_rel_));
    lag_ = std::make_shared<FabLagrangian>(lagrangian_psi, FabLagrangianArgs{{"var", relative_vars_}});
  }

  FabDifferentialMapPtr map() const override { return forward_map_; }

private:
  void set_forward_map(const std::string& goal_name) {
    auto reference_name = "x_" + goal_name;
    auto weight_name = "weight_" + goal_name;
    const auto& parent_params = parent_vars_->parameters();

    weight_var_ =
        parent_params.contains(weight_name) ? parent_params.at(weight_name) : CaSX::sym(weight_name, 1);

    geom_params_ = {{std::move(weight_name), weight_var_}};
    parent_vars_->add_parameters(geom_params_);
    // NOTE: Not working for [Particle]'s dynamic goal yet
    forward_map_ = std::make_shared<FabDifferentialMap>(forward_kinematics_, parent_vars_);
  }

protected:
  CaSX weight_var_;
  CaSX reference_var_;
  FabDifferentialMapPtr forward_map_ = nullptr;
};