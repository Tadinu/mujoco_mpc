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
#include "mjpc/planners/fabrics/include/fab_variables.h"

class FabSpectralSemiSprays {
 public:
  FabSpectralSemiSprays() = default;

  FabSpectralSemiSprays(
      CaSX M, const FabNamedMap<CaSX, FabVariables, FabTrajectories, std::vector<std::string>>& kwargs) {
    initialize(std::move(M), kwargs);
  }

  void initialize(CaSX M,
                  const FabNamedMap<CaSX, FabVariables, FabTrajectories, std::vector<std::string>>& kwargs) {
    M_ = std::move(M);
    // [x_ref_name_, xdot_ref_name_, xddot_ref_name_]
    if (kwargs.contains("ref_names")) {
      auto ref_names = *fab_core::get_arg_value<std::vector<std::string>>(kwargs, "ref_names");
      assert(ref_names.size() == 3);
      x_ref_name_ = std::move(ref_names[0]);
      xdot_ref_name_ = std::move(ref_names[1]);
      xddot_ref_name_ = std::move(ref_names[2]);
    }

    // [f_]
    if (kwargs.contains("f")) {
      f_ = *fab_core::get_arg_value<decltype(f_)>(kwargs, "f");
    }

    // [h_]
    if (kwargs.contains("h")) {
      h_ = *fab_core::get_arg_value<decltype(h_)>(kwargs, "h");
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
    }

    // [J_ref_, J_ref_inv_]
    if (is_dynamic()) {
      const auto size = x_ref().size().first;
      J_ref_ = CaSX::eye(size);
      J_ref_inv_ = CaSX::eye(size);
    } else if (kwargs.contains("J_ref")) {
      J_ref_ = *fab_core::get_arg_value<decltype(J_ref_)>(kwargs, "J_ref");
      FAB_PRINT("Casadi pseudo inverse is used in Lagrangian");
      const auto size = x_ref().size().first;
      const auto J_ref_transpose = J_ref_.T();
      J_ref_inv_ = CaSX::mtimes(J_ref_transpose,
                                CaSX::inv(CaSX::mtimes(J_ref_, J_ref_transpose) + CaSX::eye(size) * FAB_EPS));
    }

#if 0
    // [xdot_d_]
    auto xdot_d_ = CaSX::zeros(1, vars_.position_var().size().first);
#endif
  }

  std::vector<std::string> ref_names() const { return {x_ref_name_, xdot_ref_name_, xddot_ref_name_}; }

  virtual CaSX x() const { return vars_.position_var(); }

  virtual CaSX xdot() const { return vars_.velocity_var(); }
  virtual CaSX xddot() const { return xddot_; }

  FabVariables vars() const { return vars_; }

  FabTrajectories refTrajs() const { return refTrajs_; }

  CaSX M() const { return M_; }

  CaSX Minv() const {
    FAB_PRINT("Casadi pseudo inverse is used in spec");
    return CaSX::pinv(M_ + CaSX::eye(x().size().first) * FAB_EPS);
  }

  bool is_dynamic() const { return vars_.parameters().contains(x_ref_name_); }

  CaSX x_ref() const { return vars_.parameter(x_ref_name_); }

  bool has_h() const { return !h_.is_empty(); }

  CaSX h() const { return has_h() ? h_ : CaSX::mtimes(Minv(), f_); }

  bool has_f() const { return !f_.is_empty(); }

  CaSX f() const { return has_f() ? f_ : CaSX::mtimes(M(), h_); }

  bool empty() const { return !(has_f() || has_h()); }

  void concretize() {
    if (empty()) {
      return;
    }
    xddot_ = -h();
    auto vars = vars_;
    for (const auto& traj : this->refTrajs_) {
      // TODO
      vars += traj;
    }
    func_ = std::make_shared<FabCasadiFunction>("func_", vars,
                                                CaSXDict{{"M", M()}, {"f", f()}, {"xddot", xddot_}});
  }

  FabSpectralSemiSprays operator+(const FabSpectralSemiSprays& b) const {
    assert(fab_core::check_compatibility(*this, b));

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

    if (has_h() && b.has_h()) {
      return FabSpectralSemiSprays(
          M() + b.M(), {{"h", h() + b.h()},
                        {"var", all_vars},
                        {"ref_names", fab_core::get_variant_value<std::vector<std::string>>(
                                          all_ref_arguments["ref_names"])},
                        {"J_ref", fab_core::get_variant_value<CaSX>(all_ref_arguments["J_ref"])}});
    } else {
      return FabSpectralSemiSprays(
          M() + b.M(), {{"f", f() + b.f()},
                        {"var", all_vars},
                        {"ref_names", fab_core::get_variant_value<std::vector<std::string>>(
                                          all_ref_arguments["ref_names"])},
                        {"J_ref", fab_core::get_variant_value<CaSX>(all_ref_arguments["J_ref"])}});
    }
  }

  FabSpectralSemiSprays& operator+=(const FabSpectralSemiSprays& b) {
    (*this) = (*this) + b;
    return *this;
  }

  FabSpectralSemiSprays pull(const FabDifferentialMap& dm) const {
    const auto Jt = dm.J().T();
    const auto M_pulled = CaSX::mtimes(Jt, CaSX::mtimes(M(), dm.J()));

    const auto f_1 = CaSX::mtimes(Jt, CaSX::mtimes(M(), dm.Jdotqdot()));
    const auto f_2 = CaSX::mtimes(Jt, f());
    const auto f_pulled = f_1 + f_2;
    const auto x = vars_.position_var();
    const auto xdot = vars_.velocity_var();
    const auto M_pulled_subst_x = CaSX::substitute(M_pulled, x, dm.phi());

#if 0
    const auto dm_phidot = dm.phidot();
    FAB_PRINTDB("===========");
    FAB_PRINTDB("DM J");
    FAB_PRINTDB("DM VARS");
    dm.vars().print_self();
    FAB_PRINTDB(xdot, dm_phidot);
    FAB_PRINTDB(xdot.size(), dm_phidot.size());
    FAB_PRINTDB(xdot.sparsity(), dm_phidot.sparsity());
    for (auto i = 0; i < xdot.size1(); ++i) {
      FAB_PRINTDB(i, fab_core::get_casx(xdot, i), fab_core::get_casx(dm_phidot, i));
      FAB_PRINTDB("sparsity", fab_core::get_casx(xdot, i).sparsity(),
                      fab_core::get_casx(dm_phidot, i).sparsity());
      FAB_PRINTDB("dm_phidot", i, fab_core::get_casx(dm_phidot, i).is_scalar(),
                      "nnz:", fab_core::get_casx(dm_phidot, i).nnz());
    }
#endif

    auto M_pulled_subst_x_xdot = CaSX::substitute(M_pulled_subst_x, xdot, dm.phidot());

    const auto f_pulled_subst_x = CaSX::substitute(f_pulled, x, dm.phi());
    auto f_pulled_subst_x_xdot = CaSX::substitute(f_pulled_subst_x, xdot, dm.phidot());

    CaSXDict new_parameters;
    FabVariables::append_variants<CaSX>(new_parameters, vars_.parameters());
    FabVariables::append_variants<CaSX>(new_parameters, dm.parameters());
    auto new_vars = FabVariables(dm.state_variables(), new_parameters);
    auto J_ref = dm.J();
    if (is_dynamic()) {
      return FabSpectralSemiSprays(std::move(M_pulled_subst_x_xdot), {{"f", std::move(f_pulled_subst_x_xdot)},
                                                                      {"var", std::move(new_vars)},
                                                                      {"J", std::move(J_ref)},
                                                                      {"ref_names", ref_names()}});
    }
    return FabSpectralSemiSprays(
        std::move(M_pulled_subst_x_xdot),
        {{"f", std::move(f_pulled_subst_x_xdot)}, {"var", std::move(new_vars)}, {"ref_names", ref_names()}});
  }

  FabSpectralSemiSprays dynamic_pull(const FabDynamicDifferentialMap& dm) const {
    const auto M_pulled = M();
    const auto x = vars_.position_var();
    const auto xdot = vars_.velocity_var();
    const auto M_pulled_subst_x = CaSX::substitute(M_pulled, x, dm.phi());
    const auto M_pulled_subst_x_xdot = CaSX::substitute(M_pulled_subst_x, xdot, dm.phidot());
    const auto f_pulled = f() - CaSX::mtimes(M(), dm.xddot_ref());
    const auto f_pulled_subst_x = CaSX::substitute(f_pulled, x, dm.phi());
    const auto f_pulled_subst_x_xdot = CaSX::substitute(f_pulled_subst_x, xdot, dm.phidot());
    return FabSpectralSemiSprays(
        M_pulled_subst_x_xdot,
        {{"f", f_pulled_subst_x_xdot}, {"var", dm.vars()}, {"ref_names", dm.ref_names()}});
  }

  CaSXDict evaluate(const FabCasadiArgMap& kwargs) {
    if (func_) {
      auto eval = func_->evaluate(kwargs);
      return {{"M", eval["M"]}, {"f", eval["f"]}, {"xddot", eval["xddot"]}};
    }
    throw FabError::customized("FabSpectralSemiSprays evaluation failed", "Function not defined");
  }

 protected:
  std::string x_ref_name_ = "x_ref";
  std::string xdot_ref_name_ = "xdot_ref";
  std::string xddot_ref_name_ = "xddot_ref";
  FabVariables vars_;
  CaSX M_;
  CaSX f_;
  CaSX h_;
  CaSX J_ref_;
  CaSX J_ref_inv_;
  CaSX xddot_;
  std::shared_ptr<FabCasadiFunction> func_ = nullptr;
  FabTrajectories refTrajs_;
};
