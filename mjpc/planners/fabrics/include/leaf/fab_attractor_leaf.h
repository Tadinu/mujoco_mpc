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
#include "mjpc/planners/fabrics/include/leaf/fab_leaf.h"

/* FabGenericAttractorLeaf is a leaf to the tree of fabrics.
 * The attractor's potential and metric are defined through the corresponding functions to which the symbolic
 * expression function
 */
class FabGenericAttractorLeaf : public FabLeaf {
public:
  FabGenericAttractorLeaf() = default;
  ~FabGenericAttractorLeaf() override = default;

  FabGenericAttractorLeaf(const FabVariablesPtr& root_variables, const CaSX& fk_goal,
                          const std::string& attractor_name)
      : FabLeaf(attractor_name + "_leaf", root_variables, fk_goal, int(fk_goal.size().first)) {
    set_forward_map(attractor_name);
  }

  void set_potential(const FabConfigFunc& potential) override {
    const auto [x_potential, var_names] = potential(x_, xdot_, leaf_name_);
    parent_vars_->add_parameters(fab_core::parse_symbolic_casx(x_potential, var_names));
    const CaSX psi = weight_var_ * x_potential;
    CaSX h_psi = CaSX::gradient(psi, x_);
    geom_ = std::make_shared<FabGeometry>(name() + "_geom",
                                          FabGeometryArgs{{"h", std::move(h_psi)}, {"var", leaf_vars_}});
  }

  void set_metric(const FabConfigFunc& metric) override {
    const auto x = leaf_vars_->position_var();
    const auto xdot = leaf_vars_->velocity_var();
    const auto [attractor_metric, var_names] = metric(x, xdot, leaf_name_);
    parent_vars_->add_parameters(fab_core::parse_symbolic_casx(attractor_metric, var_names));
    const auto lagrangian_psi = CaSX::dot(xdot, CaSX::mtimes(attractor_metric, xdot));
    lag_ = std::make_shared<FabLagrangian>(name() + "_lag", lagrangian_psi,
                                           FabLagrangianArgs{{"var", leaf_vars_}});
  }

private:
  void set_forward_map(const std::string& goal_name) {
    auto reference_name = "x_" + goal_name;
    auto weight_name = "weight_" + goal_name;
    const auto goal_dimension = forward_kinematics_.size().first;
    const auto& parent_params = parent_vars_->parameters();
    reference_var_ = parent_params.contains(reference_name) ? parent_params.at(reference_name)
                                                            : CaSX::sym(reference_name, goal_dimension);
    weight_var_ =
        parent_params.contains(weight_name) ? parent_params.at(weight_name) : CaSX::sym(weight_name, 1);

    geom_params_ = {{std::move(reference_name), reference_var_}, {std::move(weight_name), weight_var_}};
    leaf_vars_->add_parameters(geom_params_);
    parent_vars_->add_parameters(geom_params_);
    diffmap_ = std::make_shared<FabParameterizedGoalMap>(name() + "_diffmap", parent_vars_,
                                                         forward_kinematics_, reference_var_);
  }

protected:
  CaSX weight_var_;
  CaSX reference_var_;
};
