#pragma once

#include <casadi/casadi.hpp>
#include <map>
#include <memory>
#include <stdexcept>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_energy.h"
#include "mjpc/planners/fabrics/include/fab_geometry.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"

class FabLeaf {
 public:
  FabLeaf() = default;
  virtual ~FabLeaf() = default;

  FabLeaf(FabVariables parent_variables, std::string leaf_name, CaSX fk = CaSX::zeros(), const int dim = 1)
      : parent_vars_(std::move(parent_variables)),
        leaf_name_(std::move(leaf_name)),
        x_(CaSX::sym(std::string("x_") + leaf_name_, dim)),
        xdot_(CaSX::sym(std::string("xdot_") + leaf_name_, dim)),
        leaf_vars_(FabVariables({{x_.name(), x_}, {xdot_.name(), xdot_}})),
        forward_kinematics_(std::move(fk)),
        diffmap_(std::make_shared<FabDifferentialMap>(forward_kinematics_, parent_vars_)) {}

  void set_params(const CaSXDict& kwargs) {
    for (const auto& [key, _] : p_) {
      if (kwargs.contains(key)) {
        p_.insert_or_assign(key, kwargs.at(key));
      }
    }
  }

  CaSX x() const { return x_; }
  CaSX xdot() const { return xdot_; }
  std::string name() const { return leaf_name_; }

  FabWeightedGeometry geometry() const { return geom_; }
  FabLagrangian lagrangian() const { return lag_; }
  std::shared_ptr<FabDifferentialMap> map() const { return diffmap_; }

  void concretize() {
    diffmap_->concretize();
    geom_.concretize();
    lag_.concretize();
  }

  CaSX get_parent_var_param(const std::string& var_name, const size_t dim) const {
    const auto parent_var_params = parent_vars_.parameters();
    return parent_var_params.contains(var_name) ? parent_var_params.at(var_name) : CaSX::sym(var_name, dim);
  }

  CaSXDict evaluate(const FabCasadiArgMap& kwargs) {
    const auto res = diffmap_->forward(kwargs);
    const auto x = res.at("phi");
    const auto J = res.at("J");
    const auto Jdot = res.at("Jdot");
    const auto xdot = CaSX::dot(J, std::get<CaSX>(kwargs.at("qdot")));
#if 1
    return {{"x", x}, {"xdot", xdot}};
#else
    if (geom_.empty() || lag_.empty()) {
      return CaSXDict();
    }
    const auto state_variable_names = fab_core::get_names(geom_.vars().state_variables());
    const decltype(kwargs) task_space_arguments = {{state_variable_names[0], x},
                                                   {state_variable_names[1], xdot}};
    std::copy(kwargs.begin(), kwargs.end(), std::back_inserter(task_space_arguments));
    const auto eval_geom = geom_.evaluate(task_space_arguments);
    const auto h = eval_geom.at("h");
    const auto xddot = eval_geom.at("xddot");

    const auto eval_lag = lag_.evaluate(task_space_arguments);
    const auto M = eval_geom.at("M");
    const auto f = eval_geom.at("f");
    const auto H = eval_geom.at("H");

    auto pulled_geo = geom_.pull(diffmap_);
    pulled_geo.concretize();
    const auto eval_pulled_geo = pulled_geo.evaluate(kwargs);
    const auto h_pulled = eval_pulled_geo.at("h");
    const auto xddot_pulled = eval_pulled_geo.at("xddot");

    return {{"x", x}, {"xdot", xdot}, {"h", h}, {"M", M}, {"f", f}, {"h_pulled", h_pulled}};
#endif
  }

 protected:
  FabVariables parent_vars_;
  std::string leaf_name_;
  CaSX x_;
  CaSX xdot_;
  FabVariables leaf_vars_;
  CaSX forward_kinematics_;
  FabLagrangian lag_;
  FabWeightedGeometry geom_;
  std::shared_ptr<FabDifferentialMap> diffmap_ = nullptr;
  CaSXDict p_;
};
