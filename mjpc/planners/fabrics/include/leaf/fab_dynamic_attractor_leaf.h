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

/* The GenericAttractor is a leaf to the tree of fabrics.
 * The attractor's potential and metric are defined through the corresponding functions to which the symbolic
 * expression is passed as a string.
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

  void set_potential(const std::function<CaSX(const CaSX& x, const double weight)>& potential,
                     const double weight) override {
    const CaSX psi = weight_var_ * potential(x_rel_, weight);
    CaSX h_psi = CaSX::gradient(psi, x_rel_);
    geom_ = std::make_shared<FabGeometry>(FabGeometryArgs{{"h", std::move(h_psi)}, {"var", relative_vars_}});
  }

  void set_metric(const std::function<CaSX(const CaSX& x)>& metric) override {
    const auto lagrangian_psi = CaSX::dot(xdot_rel_, CaSX::mtimes(metric(x_rel_), xdot_rel_));
    lag_ = std::make_shared<FabLagrangian>(lagrangian_psi, FabLagrangianArgs{{"var", relative_vars_}});
  }

  FabDifferentialMapPtr map() const override { return forward_map_; }

private:
  void set_forward_map(const std::string& goal_name) {
    auto reference_name = std::string("x_") + goal_name;
    auto weight_name = std::string("weight_") + goal_name;
    const auto goal_dimension = forward_kinematics_.size().first;
    const auto& parent_params = parent_vars_->parameters();
    reference_var_ = parent_params.contains(reference_name) ? parent_params.at(reference_name)
                                                            : CaSX::sym(reference_name, goal_dimension);

    weight_var_ =
        parent_params.contains(weight_name) ? parent_params.at(weight_name) : CaSX::sym(weight_name, 1);

    geom_params_ = {{std::move(reference_name), reference_var_}, {std::move(weight_name), weight_var_}};
    leaf_vars_->add_parameters(geom_params_);
    parent_vars_->add_parameters(geom_params_);
    forward_map_ =
        std::make_shared<FabParameterizedGoalMap>(parent_vars_, forward_kinematics_, reference_var_);
  }

protected:
  CaSX weight_var_;
  CaSX reference_var_;
  FabDifferentialMapPtr forward_map_ = nullptr;
};