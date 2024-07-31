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

#define LEAF_VAR_NAME(var_name) std::string(#var_name) + leaf_name_
#define CASX_SYM(var_name, dim) CaSX::sym(LEAF_VAR_NAME(var_name), dim)

class FabLeaf {
public:
  FabLeaf() = default;
  virtual ~FabLeaf() = default;

  FabLeaf(FabVariablesPtr parent_vars, std::string leaf_name, const CaSX& fk = CaSX::zeros(),
          const int dim = 1)
      : parent_vars_(std::move(parent_vars)),
        leaf_name_(std::move(leaf_name)),
        x_(CASX_SYM(x_, dim)),
        xdot_(CASX_SYM(xdot_, dim)),
        leaf_vars_(
            std::make_shared<FabVariables>(CaSXDict{{LEAF_VAR_NAME(x_), x_}, {LEAF_VAR_NAME(xdot_), xdot_}})),
        forward_kinematics_(fk),
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

  FabVariablesPtr vars() const { return leaf_vars_; }
  FabGeometryPtr geometry() const { return geom_; }
  FabLagrangianPtr lagrangian() const { return lag_; }
  virtual FabDifferentialMapPtr map() const { return diffmap_; }

  void print_self() const {
    FAB_PRINT("LEAF", name());
    vars()->print_self();
    FAB_PRINT("PARENT VARS");
    parent_vars_->print_self();
    FAB_PRINT("DIFF MAP");
    diffmap_->print_self();

    // NOTE: [geom_, lag_]'s vars may not be always available
    if (geom_ && geom_->vars()) {
      FAB_PRINT("GEOMETRY");
      geom_->vars()->print_self();
    }

    if (lag_ && lag_->vars()) {
      FAB_PRINT("LAGRANGIAN");
      lag_->vars()->print_self();
    }
    FAB_PRINT("=============");
  }

  virtual void set_potential(const std::function<CaSX(const CaSX& x, const double weight)>& potential,
                             const double weight) {}
  virtual void set_metric(const std::function<CaSX(const CaSX& x)>& metric) {}
  virtual void set_forward_map(const std::string& goal_name) {}

  void concretize() {
    diffmap_->concretize();
    geom_->concretize();
    lag_->concretize();
  }

  CaSX get_parent_var_param(const std::string& var_name, const size_t dim) const {
    const auto parent_var_params = parent_vars_->parameters();
    return parent_var_params.contains(var_name) ? parent_var_params.at(var_name)
                                                : CaSX::sym(var_name, casadi_int(dim));
  }

  CaSXDict evaluate(const FabCasadiArgMap& kwargs) {
    const auto res = diffmap_->forward(kwargs);
    const auto x = res.at("phi");
    const auto J = res.at("J");
    const auto Jdot = res.at("Jdot");
    const auto xdot = CaSX::dot(J, fab_core::get_variant_value<CaSX>(kwargs.at("qdot")));
#if 1
    return {{"x", x}, {"xdot", xdot}};
#else
    if (geom_.empty() || lag_->empty()) {
      return CaSXDict();
    }
    const auto state_variable_names = fab_core::get_names(geom_.vars()->state_variables());
    const decltype(kwargs) task_space_arguments = {{state_variable_names[0], x},
                                                   {state_variable_names[1], xdot}};
    std::copy(kwargs.begin(), kwargs.end(), std::back_inserter(task_space_arguments));
    const auto eval_geom = geom_.evaluate(task_space_arguments);
    const auto h = eval_geom.at("h");
    const auto xddot = eval_geom.at("xddot");

    const auto eval_lag = lag_->evaluate(task_space_arguments);
    const auto M = eval_geom.at("M");
    const auto f = eval_geom.at("f");
    const auto H = eval_geom.at("h");

    auto pulled_geo = geom_.pull(diffmap_);
    pulled_geo.concretize();
    const auto eval_pulled_geo = pulled_geo.evaluate(kwargs);
    const auto h_pulled = eval_pulled_geo.at("h");
    const auto xddot_pulled = eval_pulled_geo.at("xddot");

    return {{"x", x}, {"xdot", xdot}, {"h", h}, {"M", M}, {"f", f}, {"h_pulled", h_pulled}};
#endif
  }

protected:
  FabVariablesPtr parent_vars_ = nullptr;
  std::string leaf_name_;
  CaSX x_;
  CaSX xdot_;
  FabVariablesPtr leaf_vars_ = nullptr;
  CaSX forward_kinematics_;
  FabLagrangianPtr lag_ = nullptr;
  FabGeometryPtr geom_ = nullptr;
  CaSXDict geom_params_;
  FabDifferentialMapPtr diffmap_ = nullptr;
  CaSXDict p_;
};
using FabLeafPtr = std::shared_ptr<FabLeaf>;
