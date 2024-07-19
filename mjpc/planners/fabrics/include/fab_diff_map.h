#pragma once

#include <casadi/casadi.hpp>
#include <memory>

#include "mjpc/planners/fabrics/include/fab_casadi_function.h"
#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_core_util.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"

using FabDiffMapArg = FabNamedMap<uint8_t, CaSX>;

class FabDifferentialMap {
 public:
  FabDifferentialMap() = default;

  FabDifferentialMap(CaSX phi, FabVariables vars, const FabDiffMapArg& kwargs = {})
      : phi_(std::move(phi)), vars_(std::move(vars)) {
    const auto q = vars_.position_var();
    const auto qdot = vars_.velocity_var();
    J_ = CaSX::jacobian(phi_, q);
    uint8_t jdot_sign = -1;
    if (auto* jdot_sign_ptr = fab_core::get_arg_value<uint8_t>(kwargs, "Jdot_sign")) {
      jdot_sign = *jdot_sign_ptr;
    }
    Jdot_ = jdot_sign * CaSX::jacobian(CaSX::mtimes(J_, qdot), q);
  }

  CaSX J() const { return J_; }

  CaSX Jdotqdot() const { return CaSX::mtimes(Jdot_, qdot()); }

  FabVariables vars() const { return vars_; }
  CaSX q() const { return vars_.position_var(); }
  CaSX qdot() const { return vars_.velocity_var(); }
  CaSX phi() const { return phi_; }
  CaSX phidot() const { return CaSX::mtimes(J_, qdot()); }

  CaSXDict parameters() const { return vars_.parameters(); }
  CaSXDict state_variables() const { return vars_.state_variables(); }

  virtual void concretize() {
    func_ = std::make_shared<FabCasadiFunction>("func_", vars_,
                                                CaSXDict{{"phi", phi_}, {"J", J_}, {"Jdot", Jdot_}});
  }

  virtual CaSXDict forward(const FabCasadiArgMap& kwargs) {
    if (func_) {
      auto eval = func_->evaluate(kwargs);
      return {{"phi", eval["phi"]}, {"J", eval["J"]}, {"Jdot", eval["Jdot"]}};
    }
    return {};
  }

 protected:
  CaSX phi_;
  FabVariables vars_;
  CaSX J_;
  CaSX Jdot_;
  std::shared_ptr<FabCasadiFunction> func_ = nullptr;
};

using FabDifferentialMapPtr = std::shared_ptr<FabDifferentialMap>;

class FabDynamicDifferentialMap : public FabDifferentialMap {
 public:
  FabDynamicDifferentialMap() = default;

  FabDynamicDifferentialMap(const FabVariables& vars,
                            std::vector<std::string> ref_names = {"x_ref", "xdot_ref", "xddot_ref"},
                            const FabDiffMapArg& kwargs = {})
      : FabDifferentialMap(vars.position_var() - vars.parameter(ref_names[0]), vars, kwargs) {
    x_ref_name_ = std::move(ref_names[0]);
    xdot_ref_name_ = std::move(ref_names[1]);
    xddot_ref_name_ = std::move(ref_names[2]);
    phi_dot_ = vars.velocity_var() - vars.parameter(xdot_ref_name_);
  }

  virtual ~FabDynamicDifferentialMap() = default;

  CaSX x_ref() const { return this->vars_.parameter(x_ref_name_); }

  CaSX xdot_ref() const { return this->vars_.parameter(xdot_ref_name_); }

  CaSX xddot_ref() const { return this->vars_.parameter(xddot_ref_name_); }

  CaSX phidot() const { return phi_dot_; }

  std::vector<std::string> ref_names() const { return {x_ref_name_, xdot_ref_name_, xddot_ref_name_}; }

  void concretize() override {
    this->func_ = std::make_shared<FabCasadiFunction>(
        "func_", this->vars_, CaSXDict{{"x_rel", this->phi_}, {"xdot_rel", phi_dot_}});
  }

  CaSXDict forward(const FabCasadiArgMap& kwargs) override {
    if (this->func_) {
      auto eval = this->func_->evaluate(kwargs);
      return {{"x_rel", eval["x_rel"]}, {"xdot_rel", eval["xdot_rel"]}};
    }
    return {};
  }

 protected:
  std::string x_ref_name_;
  std::string xdot_ref_name_;
  std::string xddot_ref_name_;
  CaSX phi_dot_;
  CaSX Jdot_qdot_;
};

class FabExplicitDifferentialMap : public FabDifferentialMap {
 public:
  FabExplicitDifferentialMap() = default;

  FabExplicitDifferentialMap(CaSX phi, FabVariables vars, const FabDiffMapArg& kwargs)
      : FabDifferentialMap(std::move(phi), std::move(vars), kwargs) {
    this->J_ = *fab_core::get_arg_value<decltype(this->J_)>(kwargs, "J");
    this->Jdot_ = *fab_core::get_arg_value<decltype(this->Jdot_)>(kwargs, "Jdot");
  }
};
