#pragma once

#include <casadi/casadi.hpp>
#include <memory>

#include "mjpc/planners/fabrics/include/fab_casadi_function.h"
#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_core_util.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"

using FabDiffMapArg = FabNamedMap<int8_t, CaSX>;

class FabDifferentialMap {
public:
  FabDifferentialMap() = default;

  FabDifferentialMap(std::string name, const CaSX& phi, const FabVariablesPtr& vars,
                     const FabDiffMapArg& kwargs = {})
      : name_(std::move(name)), phi_(phi), vars_(vars) {
    const auto q = vars_->position_var();
    const auto qdot = vars_->velocity_var();
    J_ = CaSX::jacobian(phi_, q);
    int8_t jdot_sign = -1;
    if (auto* jdot_sign_ptr = fab_core::get_arg_value<int8_t>(kwargs, "Jdot_sign")) {
      jdot_sign = *jdot_sign_ptr;
    }
    Jdot_ = jdot_sign * CaSX::jacobian(CaSX::mtimes(J_, qdot), q);
  }

  std::string name() const { return name_; }

  CaSX J() const { return J_; }
  CaSX Jdotqdot() const { return CaSX::mtimes(Jdot_, qdot()); }

  FabVariablesPtr vars() const { return vars_; }
  CaSX q() const { return vars_->position_var(); }
  CaSX qdot() const { return vars_->velocity_var(); }
  CaSX phi() const { return phi_; }
  virtual CaSX phidot() const { return CaSX::mtimes(J_, qdot()); }

  CaSXDict parameters() const { return vars_->parameters(); }
  CaSXDict state_variables() const { return vars_->state_variables(); }

  virtual void concretize() {
    func_ = std::make_shared<FabCasadiFunction>(name() + "_func", *vars_,
                                                CaSXDict{{"phi", phi_}, {"J", J_}, {"Jdot", Jdot_}});
  }

  virtual CaSXDict forward(const FabCasadiArgMap& kwargs) {
    if (func_) {
      auto eval = func_->evaluate(kwargs);
      return {{"phi", eval["phi"]}, {"J", eval["J"]}, {"Jdot", eval["Jdot"]}};
    }
    return {};
  }

  void print_self() const {
    FAB_PRINT("VARS");
    vars_->print_self();
    FAB_PRINT("q", q(), q().size());
    FAB_PRINT("qdot", qdot(), qdot().size());
    FAB_PRINT("phi", phi(), phi().size());
    FAB_PRINT("J", J(), J().size());
    FAB_PRINT("Jdot", Jdot_);
  }

protected:
  std::string name_;
  CaSX phi_;
  FabVariablesPtr vars_ = nullptr;
  CaSX J_;
  CaSX Jdot_;
  std::shared_ptr<FabCasadiFunction> func_ = nullptr;
};

using FabDifferentialMapPtr = std::shared_ptr<FabDifferentialMap>;

class FabDynamicDifferentialMap : public FabDifferentialMap {
public:
  FabDynamicDifferentialMap() = default;

  explicit FabDynamicDifferentialMap(std::string name, const FabVariablesPtr& vars,
                                     std::array<std::string, 3> ref_names = {"x_ref", "xdot_ref",
                                                                             "xddot_ref"},
                                     const FabDiffMapArg& kwargs = {})
      : FabDifferentialMap(std::move(name), vars->position_var() - vars->parameter(ref_names[0]), vars,
                           kwargs) {
    x_ref_name_ = std::move(ref_names[0]);
    xdot_ref_name_ = std::move(ref_names[1]);
    xddot_ref_name_ = std::move(ref_names[2]);
    phi_dot_ = vars->velocity_var() - vars->parameter(xdot_ref_name_);
  }

  virtual ~FabDynamicDifferentialMap() = default;

  CaSX x_ref() const { return this->vars_->parameter(x_ref_name_); }

  CaSX xdot_ref() const { return this->vars_->parameter(xdot_ref_name_); }

  CaSX xddot_ref() const { return this->vars_->parameter(xddot_ref_name_); }

  CaSX phidot() const override { return phi_dot_; }

  std::vector<std::string> ref_names() const { return {x_ref_name_, xdot_ref_name_, xddot_ref_name_}; }

  void concretize() override {
    this->func_ = std::make_shared<FabCasadiFunction>(
        name_ + "_func", *this->vars_, CaSXDict{{"x_rel", this->phi_}, {"xdot_rel", phi_dot_}});
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
  // CaSX Jdot_qdot_;
};

using FabDynamicDifferentialMapPtr = std::shared_ptr<FabDynamicDifferentialMap>;

class FabExplicitDifferentialMap : public FabDifferentialMap {
public:
  FabExplicitDifferentialMap() = default;

  FabExplicitDifferentialMap(std::string name, const CaSX& phi, const FabVariablesPtr& vars,
                             const FabDiffMapArg& kwargs)
      : FabDifferentialMap(std::move(name), phi, vars, kwargs) {
    this->J_ = *fab_core::get_arg_value<decltype(this->J_)>(kwargs, "J");
    this->Jdot_ = *fab_core::get_arg_value<decltype(this->Jdot_)>(kwargs, "Jdot");
  }
};
