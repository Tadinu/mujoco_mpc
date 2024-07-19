#pragma once

#include <casadi/casadi.hpp>
#include <map>
#include <memory>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_energy.h"
#include "mjpc/planners/fabrics/include/fab_parameterized_map.h"
#include "mjpc/planners/fabrics/include/fab_config.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"
#include "mjpc/planners/fabrics/include/leaf/fab_dynamic_leaf.h"

/* The GenericAttractor is a leaf to the tree of fabrics.
 * The attractor's potential and metric are defined through the corresponding functions to which the symbolic
 * expression is passed as a string.
 */
class FabGenericDynamicAttractorLeaf : public FabDynamicLeaf {
 public:
  FabGenericDynamicAttractorLeaf() = default;

  FabGenericDynamicAttractorLeaf(FabVariables root_variables, CaSX fk_goal, const std::string& attractor_name)
      : FabDynamicLeaf(std::move(root_variables), attractor_name + "_leaf", fk_goal.size().first,
                       fk_goal.size().first, std::move(fk_goal)) {
    set_forward_map(attractor_name);
  }

  void set_forward_map(const std::string& goal_name) override {
    const auto weight_name = std::string("weight_") + goal_name;
    const auto& parent_params = parent_vars_.parameters();
    if (parent_params.contains(weight_name)) {
      weight_var_ = parent_params.at(weight_name);
    } else {
      weight_var_ = CaSX::sym(weight_name, 1);
    }
    geom_params_ = {{weight_name, weight_var_}};
    parent_vars_.add_parameters(geom_params_);
    forward_map_ = std::make_shared<FabDifferentialMap>(forward_kinematics_, parent_vars_);
  }

  void set_potential(const std::function<CaSX(const CaSX& x)>& potential) override {
    const auto psi = weight_var_ * potential(x_rel_);
    geom_ = FabGeometry({{"h", CaSX::gradient(psi, x_rel_)}, {"var", relative_vars_}});
  }

  void set_metric(const std::function<CaSX(const CaSX& x)>& metric) override {
    auto lagrangian_psi = CaSX::dot(xdot_rel_, CaSX::mtimes(metric(x_rel_), xdot_rel_));
    lag_ = FabLagrangian(std::move(lagrangian_psi), {{"var", relative_vars_}});
  }

  virtual FabDifferentialMapPtr map() const override { return forward_map_; }

 protected:
  CaSX weight_var_;
  std::shared_ptr<FabDifferentialMap> forward_map_;
};