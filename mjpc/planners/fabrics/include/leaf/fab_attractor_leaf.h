#pragma once

#include <casadi/casadi.hpp>
#include <map>
#include <memory>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_energy.h"
#include "mjpc/planners/fabrics/include/fab_parameterized_map.h"
#include "mjpc/planners/fabrics/include/fab_planner_config.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"
#include "mjpc/planners/fabrics/include/leaf/fab_leaf.h"

/* The GenericAttractor is a leaf to the tree of fabrics.
 * The attractor's potential and metric are defined through the corresponding functions to which the symbolic
 * expression is passed as a string.
 */
class FabGenericAttractorLeaf : public FabLeaf {
 public:
  FabGenericAttractorLeaf() = default;
  virtual ~FabGenericAttractorLeaf() = default;

  FabGenericAttractorLeaf(FabVariables root_variables, CaSX fk_goal, const std::string& attractor_name)
      : FabLeaf(std::move(root_variables), attractor_name + "_leaf", std::move(fk_goal),
                fk_goal.size().first) {
    set_forward_map(attractor_name);
  }

  void set_forward_map(const std::string& goal_name) {
    const auto reference_name = std::string("x_") + goal_name;
    const auto weight_name = std::string("weight_") + goal_name;
    const int goal_dimension = forward_kinematics_.size().first;
    const auto& parent_params = parent_vars_.parameters();
    reference_var_ = parent_params.contains(reference_name) ? parent_params.at(reference_name)
                                                            : CaSX::sym(reference_name, goal_dimension);
    weight_var_ =
        parent_params.contains(reference_name) ? parent_params.at(weight_name) : CaSX::sym(weight_name, 1);

    geom_params_ = {{reference_name, reference_var_}, {weight_name, weight_var_}};
    leaf_vars_.add_parameters(geom_params_);
    parent_vars_.add_parameters(geom_params_);
    diffmap_ = std::make_shared<FabParameterizedGoalMap>(parent_vars_, forward_kinematics_, reference_var_);
  }

  void set_potential(const std::function<CaSX(const CaSX& x)>& potential) {
    // new_parameters, potential = parse_symbolic_input(potential_expression, x, xdot, name = self._leaf_name)
    // parent_vars_.add_parameters(new_parameters);
    const CaSX psi = weight_var_ * potential(x_);
    const CaSX h_psi = CaSX::gradient(psi, x_);
    geom_ = FabWeightedGeometry({{"h", h_psi}, {"var", leaf_vars_}});
  }

  void set_metric(const std::function<CaSX(const CaSX& x)>& metric) {
    const auto x = leaf_vars_.position_var();
    const auto xdot = leaf_vars_.velocity_var();
    // new_parameters, attractor_metric = parse_symbolic_input(attractor_metric_expression, x, xdot,
    // name=self._leaf_name) self._parent_variables.add_parameters(new_parameters)
    auto lagrangian_psi = CaSX::dot(xdot, CaSX::mtimes(metric(x), xdot));
    lag_ = FabLagrangian(std::move(lagrangian_psi), {{"var", leaf_vars_}});
  }

 protected:
  FabLagrangian lag_;
  CaSXDict geom_params_;
  CaSX weight_var_;
  CaSX reference_var_;
};