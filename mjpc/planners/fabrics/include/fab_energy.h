#pragma once

#include <casadi/casadi.hpp>
#include <casadi/core/generic_matrix.hpp>
#include <cassert>
#include <map>
#include <memory>
#include <stdexcept>
#include <typeinfo>

#include "mjpc/planners/fabrics/include/fab_casadi_function.h"
#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_core_util.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_spectral_semi_sprays.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"

class FabLagrangian {
 public:
  FabLagrangian() = default;

  FabLagrangian(CaSX l, const FabNamedMap<CaSX, FabVariables, FabTrajectories, FabSpectralSemiSprays,
                                          std::vector<std::string>>& kwargs)
      : l_(std::move(l)) {
    // [x_ref_name_, xdot_ref_name_, xddot_ref_name_]
    if (kwargs.contains("ref_names")) {
      const auto ref_names = *fab_core::get_arg_value<std::vector<std::string>>(kwargs, "ref_names");
      x_ref_name_ = ref_names[0];
      xdot_ref_name_ = ref_names[1];
      xddot_ref_name_ = ref_names[2];
    }

    // [vars_]
    if (kwargs.contains("x")) {
      assert(kwargs.contains("xdot"));
      vars_ = FabVariables({{"x", *fab_core::get_arg_value<CaSX>(kwargs, "x")},
                            {"xdot", *fab_core::get_arg_value<CaSX>(kwargs, "xdot")}});
    } else if (kwargs.contains("var")) {
      vars_ = *fab_core::get_arg_value<decltype(vars_)>(kwargs, "var");
    }

    // [refTrajs_]
    if (kwargs.contains("refTrajs")) {
      refTrajs_ = *fab_core::get_arg_value<decltype(refTrajs_)>(kwargs, "refTrajs");
      rel_ = refTrajs_.size() > 0;
    }

    // [J_ref_, J_ref_inv_]
    if (kwargs.contains("J_ref")) {
      J_ref_ = *fab_core::get_arg_value<decltype(J_ref_)>(kwargs, "J_ref");
      std::cout << "Casadi pseudo inverse is used in Lagrangian" << std::endl;
      const auto J_ref_transpose = J_ref_.T();
      J_ref_inv_ = CaSX::mtimes(
          J_ref_transpose,
          CaSX::inv(CaSX::mtimes(J_ref_, J_ref_transpose + CaSX::eye(x_ref().size().first) * FAB_EPS)));
    }

    // [S_, H_]
    if (!is_dynamic() && kwargs.contains("spec") && kwargs.contains("hamiltonian")) {
      H_ = *fab_core::get_arg_value<decltype(H_)>(kwargs, "hamiltonian");
      S_ = *fab_core::get_arg_value<decltype(S_)>(kwargs, "spec");
    } else {
      applyEulerLagrange();
    }
  }

  FabVariables vars() const { return vars_; }

  CaSX x() const { return vars_.position_var(); }

  CaSX xdot() const { return vars_.velocity_var(); }

  CaSX x_ref() const { return vars_.parameter(x_ref_name_); }

  CaSX xdot_ref() const { return vars_.parameter(xdot_ref_name_); }

  CaSX xddot_ref() const { return vars_.parameter(xddot_ref_name_); }

  CaSX xdot_rel(int8_t ref_sign = 1) {
    if (is_dynamic()) {
      return xdot() - ref_sign * CaSX::mtimes(J_ref_inv_, xdot_ref());
    } else {
      return xdot();
    }
  }

  FabSpectralSemiSprays S() const { return S_; }
  bool empty() const { return S_.empty(); }

  CaSX l() const { return l_; }

  FabTrajectories refTrajs() const { return refTrajs_; }

  bool is_dynamic() const { return vars_.parameters().contains(x_ref_name_); }

  std::vector<std::string> ref_names() const { return {x_ref_name_, xdot_ref_name_, xddot_ref_name_}; }

  FabTrajectories refTraj() const { return refTrajs_; }

  FabLagrangian operator+(const FabLagrangian& b) const {
    assert(fab_core::check_compatibility(*this, b));
    // const auto refTrajs = FabVariables::join_refTrajs(refTrajs_, b.refTrajs_);

    // [all+vars]
    const auto all_vars = vars_ + b.vars();

    // [all_ref_names, J_ref]
    std::vector<std::string> all_ref_names;
    CaSX J_ref;
    if (is_dynamic()) {
      const auto _ref_names = ref_names();
      std::copy(_ref_names.begin(), _ref_names.end(), std::back_inserter(all_ref_names));
      J_ref = J_ref_;
    }

    if (b.is_dynamic()) {
      const auto _ref_names = b.ref_names();
      std::copy(_ref_names.begin(), _ref_names.end(), std::back_inserter(all_ref_names));
      J_ref = b.J_ref_;
    }

    if (all_ref_names.empty()) {
      all_ref_names = ref_names();
    }

    // [ref_arguments]
    FabNamedMap<std::vector<std::string>, CaSX> all_ref_arguments;
    if (!all_ref_names.empty()) {
      all_ref_arguments = {{"ref_names", all_ref_names}, {"J_ref", J_ref}};
    }

    return FabLagrangian(l_ + b.l_,
                         {{"spec", S_ + b.S_},
                          {"hamiltonian", H_ + b.H_},
                          {"var", all_vars},
                          {"ref_names", std::get<std::vector<std::string>>(all_ref_arguments["ref_names"])},
                          {"J_ref", std::get<CaSX>(all_ref_arguments["J_ref"])}});
  }

  FabLagrangian& operator+=(const FabLagrangian& b) {
    (*this) = (*this) + b;
    return *this;
  }

  void applyEulerLagrange() {
    const auto dL_dxdot = CaSX::gradient(l_, xdot());
    const auto dL_dx = CaSX::gradient(l_, x());
    const auto d2L_dxdxdot = CaSX::jacobian(dL_dx, xdot());
    const auto d2L_dxdot2 = CaSX::jacobian(dL_dxdot, xdot());
    auto f_rel = CaSX::zeros(x().size().first);
    auto en_rel = CaSX::zeros(1);

    if (is_dynamic()) {
      const auto _x_ref = x_ref();
      const auto _xdot_ref = xdot_ref();
      const auto _xddot_ref = xddot_ref();
      const auto dL_dxpdot = CaSX::gradient(l_, _xdot_ref);
      const auto d2L_dxdotdxpdot = CaSX::jacobian(dL_dxdot, _xdot_ref);
      const auto d2L_dxdotdxp = CaSX::jacobian(dL_dxdot, _x_ref);
      const auto f_rel1 = CaSX::mtimes(d2L_dxdotdxpdot, _xddot_ref);
      const auto f_rel2 = CaSX::mtimes(d2L_dxdotdxp, _xdot_ref);
      f_rel += f_rel1 + f_rel2;
      en_rel += CaSX::dot(dL_dxpdot, _xdot_ref);

      const auto F = d2L_dxdxdot;
      const auto M = d2L_dxdot2;
      const auto f_e = -dL_dx;
      const auto f = CaSX::mtimes(F.T(), xdot()) + f_e + f_rel;
      H_ = CaSX::dot(dL_dxdot, xdot()) - l_ + en_rel;
      S_ = decltype(S_)(M, {{"f", f}, {"var", vars_}, {"refTrajs", refTrajs_}});
    }
  }

  void concretize() {
    if (empty()) {
      return;
    }
    S_.concretize();
    auto vars = vars_;
    for (const auto& refTraj : refTrajs_) {
      // TODO
      vars += refTraj;
    }

    func_ = std::make_shared<FabCasadiFunction>("func_", vars, CaSXDict{{"H", H_}});
  }

  FabLagrangian pull(const FabDifferentialMap& dm) const {
    const auto l_subst = CaSX::substitute(l_, x(), dm.phi());
    const auto l_subst2 = CaSX::substitute(l_subst, xdot(), dm.phidot());

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
    if (is_dynamic()) {
      return FabLagrangian(l_subst2, {{"var", new_vars}, {"J_ref", dm.J()}, {"ref_nanes", ref_names()}});
    } else {
      return FabLagrangian(l_subst2, {{"var", new_vars}, {"ref_nanes", ref_names()}});
    }
  }

  CaSXDict evaluate(const FabCasadiArgMap& kwargs) {
    if (func_) {
      auto H = func_->evaluate(kwargs)["H"];
      auto M_f = S_.evaluate(kwargs);
      return {{"M", M_f["M"]}, {"f", M_f["f"]}, {"H", H}};
    }
    throw FabError::customized("FabGeometry evaluation failed", "Function not defined");
  }

  FabLagrangian dynamic_pull(const FabDynamicDifferentialMap& dm) const {
    const auto& l_pulled = l_;
    const auto l_pulled_subst_x = CaSX::substitute(l_pulled, x(), dm.phi());
    const auto l_pulled_subst_x_xdot = CaSX::substitute(l_pulled_subst_x, xdot(), dm.phidot());
    return FabLagrangian(l_pulled_subst_x_xdot, {{"var", dm.vars()}, {"ref_names", dm.ref_names()}});
  }

 protected:
  CaSX l_;
  std::string x_ref_name_ = "x_ref";
  std::string xdot_ref_name_ = "xdot_ref";
  std::string xddot_ref_name_ = "xddot_ref";
  FabVariables vars_;
  CaSX H_;
  CaSX J_ref_;
  CaSX J_ref_inv_;
  std::shared_ptr<FabCasadiFunction> func_ = nullptr;
  bool rel_ = false;
  FabTrajectories refTrajs_;
  FabTrajectory refTraj_;
  FabSpectralSemiSprays S_;
};

// -----------------------------------------------------------------------------
// FINSLER STRUCTURE --
//
class FabFinslerStructure : public FabLagrangian {
  FabFinslerStructure() = default;

  FabFinslerStructure(CaSX lg, const FabNamedMap<CaSX, FabVariables, FabTrajectories, FabSpectralSemiSprays,
                                                 std::vector<std::string>>& kwargs)
      : FabLagrangian(0.5 * pow(lg, 2), kwargs), lg_(std::move(lg)) {}

  void concretize() {
    FabLagrangian::concretize();
    func_lg_ = std::make_shared<FabCasadiFunction>("func_lg_", this->vars_, CaSXDict{{"Lg", lg_}});
  }

  inline CaSXDict evaluate(const FabCasadiArgMap& kwargs) {
    auto parent_eval = FabLagrangian::evaluate(kwargs);
    if (func_lg_) {
      return {{"M", parent_eval["M"]},
              {"f", parent_eval["f"]},
              {"h", parent_eval["H"]},
              {"lg", func_lg_->evaluate(kwargs)["Lg"]}};
    }
    throw FabError::customized("FabGeometry evaluation failed", "Function not defined");
  }

 protected:
  CaSX lg_;
  std::shared_ptr<FabCasadiFunction> func_lg_ = nullptr;
};

// -----------------------------------------------------------------------------
// EXECUTION LAGRANGIAN --
//
class FabExecutionLagrangian : public FabLagrangian {
 public:
  explicit FabExecutionLagrangian(FabVariables vars)
      : FabLagrangian(CaSX::dot(vars.velocity_var(), vars.velocity_var()), {{"var", std::move(vars)}}) {}
};
