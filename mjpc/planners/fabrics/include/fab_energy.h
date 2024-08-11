#pragma once

#include <casadi/casadi.hpp>
#include <casadi/core/generic_matrix.hpp>
#include <cassert>
#include <map>
#include <memory>
#include <typeinfo>

#include "mjpc/planners/fabrics/include/fab_casadi_function.h"
#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_core_util.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_geometry.h"
#include "mjpc/planners/fabrics/include/fab_spectral_semi_sprays.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"

using FabLagrangianArgs =
    FabNamedMap<CaSX, FabVariablesPtr, FabTrajectories, FabSpectralSemiSpraysPtr, std::vector<std::string>>;

class FabLagrangian : public FabGeometry {
public:
  FabLagrangian() = default;

  FabLagrangian(const CaSX& lag, const FabLagrangianArgs& kwargs) : l_(lag) {
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
      vars_ =
          std::make_shared<FabVariables>(CaSXDict{{"x", *fab_core::get_arg_value<CaSX>(kwargs, "x")},
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
      FAB_PRINT("Casadi pseudo inverse is used in Lagrangian");
      const auto J_ref_transpose = J_ref_.T();
      J_ref_inv_ = CaSX::mtimes(J_ref_transpose, CaSX::inv(CaSX::mtimes(J_ref_, J_ref_transpose) +
                                                           fab_math::CASX_IDENTITY(x_ref().size().first) * FAB_EPS));
    }

    // [S_, H_]
    if (!is_dynamic() && kwargs.contains("spec") && kwargs.contains("hamiltonian")) {
      h_ = *fab_core::get_arg_value<decltype(h_)>(kwargs, "hamiltonian");
      s_ = *fab_core::get_arg_value<decltype(s_)>(kwargs, "spec");
    } else {
      apply_euler_lagrange();
    }
  }

  CaSX xdot_rel(int8_t ref_sign = 1) {
    if (is_dynamic()) {
      return xdot() - ref_sign * CaSX::mtimes(J_ref_inv_, xdot_ref());
    } else {
      return xdot();
    }
  }

  FabSpectralSemiSpraysPtr s() const { return s_; }
  bool empty() const { return s_->empty(); }

  CaSX l() const { return l_; }

  FabLagrangian operator+(const FabLagrangian& b) const {
    assert(fab_core::check_compatibility(*this, b));
    // const auto refTrajs = FabVariables::join_refTrajs(refTrajs_, b.refTrajs_);

    // [all+vars]
    auto all_vars = std::make_shared<FabVariables>(*vars_ + *b.vars());

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

    return FabLagrangian(
        l_ + b.l_,
        {{"spec", std::make_shared<FabSpectralSemiSprays>(*s_ + *b.s_)},
         {"hamiltonian", h_ + b.h_},
         {"var", std::move(all_vars)},
         {"ref_names", fab_core::get_variant_value<std::vector<std::string>>(all_ref_arguments["ref_names"])},
         {"J_ref", fab_core::get_variant_value<CaSX>(all_ref_arguments["J_ref"])}});
  }

  FabLagrangian& operator+=(const FabLagrangian& b) {
    (*this) = (*this) + b;
    return *this;
  }

  void apply_euler_lagrange() {
    const bool is_scalar = l_.is_scalar();
    const auto dL_dx = is_scalar ? CaSX::gradient(l_, x()) : CaSX::jacobian(l_, x());
    const auto dL_dxdot = is_scalar ? CaSX::gradient(l_, xdot()) : CaSX::jacobian(l_, xdot());
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
    }
    const auto& F = d2L_dxdxdot;
    const auto& M = d2L_dxdot2;
    const auto f_e = -dL_dx;
    FAB_PRINTDB(F.T(), F.T().size());
    FAB_PRINTDB(xdot(), xdot().size());
    const auto f = CaSX::mtimes(F.T(), xdot()) + f_e + f_rel;
    h_ = CaSX::dot(dL_dxdot, xdot()) - l_ + en_rel;
    s_ = std::make_shared<FabSpectralSemiSprays>(
        M, FabSpecArgs{{"f", f}, {"var", vars_}, {"refTrajs", refTrajs_}});
  }

  void concretize(int8_t ref_sign = 1) override {
    if (empty()) {
      return;
    }
    s_->concretize();
    auto vars = *vars_;
    for (const auto& refTraj : refTrajs_) {
      // TODO
      vars += refTraj;
    }

    func_ = std::make_shared<FabCasadiFunction>("func_", std::move(vars), CaSXDict{{"h", h_}});
  }

protected:
  FabGeometryPtr pull(const FabDifferentialMap& dm) const override {
    const auto l_subst = CaSX::substitute(l_, x(), dm.phi());
    const auto l_subst2 = CaSX::substitute(l_subst, xdot(), dm.phidot());

    CaSXDict new_parameters;
    FabVariables::append_variants<CaSX>(new_parameters, vars_->parameters(), true);
    FabVariables::append_variants<CaSX>(new_parameters, dm.parameters(), true);
    auto new_vars = std::make_shared<FabVariables>(dm.state_variables(), new_parameters);

    FabTrajectories refTrajs;
    if (!refTraj_.empty()) {
      refTrajs.push_back(refTraj_);
    }
    for (const auto& traj : refTrajs_) {
      // TODO
      // refTrajs.push_back(traj.pull(dm));
    }
    if (is_dynamic()) {
      return std::make_shared<FabLagrangian>(
          l_subst2,
          FabLagrangianArgs{{"var", std::move(new_vars)}, {"J_ref", dm.J()}, {"ref_names", ref_names()}});
    } else {
      return std::make_shared<FabLagrangian>(
          l_subst2, FabLagrangianArgs{{"var", std::move(new_vars)}, {"ref_names", ref_names()}});
    }
  }

  // Transformation of a relative spec (Xrel) into the static space X
  FabGeometryPtr dynamic_pull(const FabDynamicDifferentialMap& dm) const override {
    const auto& l_pulled = l_;
    const auto l_pulled_subst_x = CaSX::substitute(l_pulled, x(), dm.phi());
    const auto l_pulled_subst_x_xdot = CaSX::substitute(l_pulled_subst_x, xdot(), dm.phidot());
    return std::make_shared<FabLagrangian>(
        l_pulled_subst_x_xdot, FabLagrangianArgs{{"var", dm.vars()}, {"ref_names", dm.ref_names()}});
  }

public:
  CaSXDict evaluate(const FabCasadiArgMap& kwargs) const override {
    if (func_) {
      auto eval = func_->evaluate(kwargs);
      auto M_f = s_->evaluate(kwargs);
      return {{"M", M_f["M"]}, {"f", M_f["f"]}, {"h", eval["h"]}};
    }
    throw FabError::customized("FabGeometry evaluation failed", "Function not defined");
  }

protected:
  CaSX l_;
  CaSX J_ref_;
  CaSX J_ref_inv_;
  bool rel_ = false;
  FabSpectralSemiSpraysPtr s_ = nullptr;
};

using FabLagrangianPtr = std::shared_ptr<FabLagrangian>;

// -----------------------------------------------------------------------------
// FINSLER STRUCTURE --
//
class FabFinslerStructure : public FabLagrangian {
  FabFinslerStructure() = default;

  FabFinslerStructure(const CaSX& lg,
                      const FabNamedMap<CaSX, FabVariablesPtr, FabTrajectories, FabSpectralSemiSpraysPtr,
                                        std::vector<std::string>>& kwargs)
      : FabLagrangian(0.5 * pow(lg, 2), kwargs), lg_(lg) {}

  void concretize() {
    FabLagrangian::concretize();
    func_lg_ = std::make_shared<FabCasadiFunction>("func_lg_", *this->vars_, CaSXDict{{"Lg", lg_}});
  }

  CaSXDict evaluate(const FabCasadiArgMap& kwargs) const override {
    auto parent_eval = FabLagrangian::evaluate(kwargs);
    if (func_lg_) {
      return {{"M", parent_eval["M"]},
              {"f", parent_eval["f"]},
              {"h", parent_eval["h"]},
              {"lg", func_lg_->evaluate(kwargs)["lg"]}};
    }
    throw FabError::customized("FabGeometry evaluation failed", "Function not defined");
  }

protected:
  CaSX lg_;
  std::shared_ptr<FabCasadiFunction> func_lg_ = nullptr;
};

using FabFinslerStructurePtr = std::shared_ptr<FabFinslerStructure>;

// -----------------------------------------------------------------------------
// EXECUTION LAGRANGIAN --
//
class FabExecutionLagrangian : public FabLagrangian {
public:
  explicit FabExecutionLagrangian(const FabVariablesPtr& vars)
      : FabLagrangian(CaSX::dot(vars->velocity_var(), vars->velocity_var()), {{"var", vars}}) {}
};

using FabExecutionLagrangianPtr = std::shared_ptr<FabExecutionLagrangian>;
