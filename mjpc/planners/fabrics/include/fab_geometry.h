#pragma once

#include <casadi/casadi.hpp>
#include <cassert>
#include <map>
#include <memory>
#include <stdexcept>
#include <typeinfo>

#include "mjpc/planners/fabrics/include/fab_casadi_function.h"
#include "mjpc/planners/fabrics/include/fab_spectral_semi_sprays.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"

class FabGeometry {
 public:
  FabGeometry() = default;

  explicit FabGeometry(const FabNamedMap<CaSX, FabVariables, FabTrajectories, FabSpectralSemiSprays,
                                         std::vector<std::string>>& kwargs) {
    // [h_, vars_]
    if (kwargs.contains("x")) {
      assert(kwargs.contains("xdot"));
      h_ = *fab_core::get_arg_value<decltype(h_)>(kwargs, "h");
      vars_ = FabVariables({{"x", *fab_core::get_arg_value<CaSX>(kwargs, "x")},
                            {"xdot", *fab_core::get_arg_value<CaSX>(kwargs, "xdot")}});
    } else if (kwargs.contains("var")) {
      h_ = *fab_core::get_arg_value<decltype(h_)>(kwargs, "h");
      vars_ = *fab_core::get_arg_value<decltype(vars_)>(kwargs, "var");
    } else if (kwargs.contains("s")) {
      const auto s = *fab_core::get_arg_value<FabSpectralSemiSprays>(kwargs, "s");
      h_ = s.h();  // NOTE: this uses s.vars()
      vars_ = s.vars();
    }

    // [refTrajs_]
    if (kwargs.contains("refTrajs")) {
      refTrajs_ = *fab_core::get_arg_value<decltype(refTrajs_)>(kwargs, "refTrajs");
    }
  }

  CaSX x() const { return vars_.position_var(); }

  CaSX xdot() const { return vars_.velocity_var(); }

  FabVariables vars() const { return vars_; }

  std::vector<std::string> ref_names() const { return {x_ref_name_, xdot_ref_name_, xddot_ref_name_}; }

  FabTrajectories refTrajs() const { return refTrajs_; }

  CaSX h() const { return h_; }

  FabGeometry operator+(const FabGeometry& b) const { return FabGeometry(*this) += b; }

  FabGeometry& operator+=(const FabGeometry& b) {
    assert(fab_core::check_compatibility(*this, b));
    h_ = h_ + b.h_;
    vars_ = vars_ + b.vars_;
    return *this;
  }

  FabGeometry pull(const FabDifferentialMap& dm) const {
    const auto h_pulled = CaSX::mtimes(CaSX::pinv(dm.J()), h_ + dm.Jdotqdot());
    const auto h_pulled_subst_x = CaSX::substitute(h_pulled, x(), dm.phi());
    const auto h_pulled_subst_x_xdot = CaSX::substitute(h_pulled_subst_x, xdot(), dm.phidot());

    CaSXDict new_parameters;
    FabVariables::append_variants<CaSX>(new_parameters, vars_.parameters());
    FabVariables::append_variants<CaSX>(new_parameters, dm.parameters());
    const auto new_vars = FabVariables(dm.state_variables(), new_parameters);

    FabTrajectories refTrajs;
    if (!refTraj_.empty()) {
      refTrajs.push_back(refTraj_);
    }
    for (const auto& traj : refTrajs_) {
      // TODO
      // refTrajs.push_back(traj.pull(dm));
    }
    return FabGeometry({{"h", h_pulled_subst_x_xdot}, {"var", new_vars}, {"refTrajs", refTrajs}});
  }

  FabGeometry dynamic_pull(const FabDynamicDifferentialMap& dm) {
    const auto h_pulled = h_ - dm.xddot_ref();
    const auto h_pulled_subst_x = CaSX::substitute(h_pulled, x(), dm.phi());
    const auto h_pulled_subst_x_xdot = CaSX::substitute(h_pulled_subst_x, xdot(), dm.phidot());
    return FabGeometry({{"h", h_pulled_subst_x_xdot}, {"var", dm.vars()}});
  }

  void concretize() {
    xddot_ = -h_;
    auto vars = vars_;
    for (const auto& refTraj : refTrajs_) {
      // TODO
      vars += refTraj;
    }

    func_ = std::make_shared<FabCasadiFunction>("func_", vars, CaSXDict{{"H", h_}, {"xddot", xddot_}});
  }

  CaSXDict evaluate(const FabCasadiArgMap& kwargs) const {
    if (func_) {
      auto eval = func_->evaluate(kwargs);
      return {{"h", eval["h"]}, {"xddot", eval["xddot"]}};
    }
    throw FabError::customized("FabGeometry evaluation failed", "Function not defined");
  }

  bool is_homogeneous_degree2() const {
    const auto rand_x = CaSX::rand(x().size().first);
    const auto rand_xdot = CaSX::rand(xdot().size().first);
    constexpr double alpha = 2.0;
    const auto rand_xdot2 = alpha * rand_xdot;
    const auto h = evaluate({{"x_obst_dynamic", rand_x}, {"xdot_obst_dynamic", rand_xdot}})["h"];
    const auto h2 = evaluate({{"x_obst_dynamic", rand_x}, {"xdot_obst_dynamic", rand_xdot2}})["h"];
    // return (h * pow(alpha, 2)) == h2;
    return false;
  }

 protected:
  std::string x_ref_name_ = "x_ref";
  std::string xdot_ref_name_ = "xdot_ref";
  std::string xddot_ref_name_ = "xddot_ref";
  FabVariables vars_;
  CaSX h_;
  CaSX xddot_;
  std::shared_ptr<FabCasadiFunction> func_ = nullptr;
  FabTrajectories refTrajs_;
  FabTrajectory refTraj_;
};