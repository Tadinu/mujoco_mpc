#pragma once
#include <casadi/casadi.hpp>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_energy.h"

class FabDamper {
public:
  FabDamper() = default;

  FabDamper(const CaSX& x, const FabConfigExprMeta& beta_meta, const FabConfigExprMeta& eta_meta,
            FabDifferentialMapPtr dm, const CaSX& lagrangian_execution)
      : x_(x), beta_(beta_meta.eval), eta_(eta_meta.eval), dm_(std::move(dm)), lg_(lagrangian_execution) {
    // [beta_](initially as [beta] in initializer list)
    for (const auto& beta_param : CaSX::symvar(beta_)) {
      const auto beta_param_name = beta_param.name();
      if (fab_core::has_collection_element(beta_meta.var_names, beta_param_name)) {
        symbolic_params_.insert_or_assign(beta_param_name, beta_param);
      }

      // [a_ex, a_le]
      if (beta_param_name.starts_with("a_ex")) {
        a_ex_ = beta_param;
        constant_beta_expression_ = false;
      } else if (beta_param_name.starts_with("a_le")) {
        a_le_ = beta_param;
        constant_beta_expression_ = false;
      }
    }

    // [eta_] (initially as [eta] in initializer list)
    CaSX ex_lag_sym;
    for (const auto& eta_param : CaSX::symvar(eta_)) {
      const std::string eta_param_name = eta_param.name();
      if (fab_core::has_collection_element(eta_meta.var_names, eta_param_name)) {
        symbolic_params_.insert_or_assign(eta_param_name, eta_param);
      }

      // [ex_lag]
      if (eta_param_name.starts_with("ex_lag")) {
        ex_lag_sym = eta_param;
      }
    }

    // [ex_lag_sym] -> [lagrangian_execution]
    if (!ex_lag_sym.is_empty()) {
      eta_ = CaSX::substitute(eta_, ex_lag_sym, lagrangian_execution);
    }
  }

  CaSX substitute_beta(const CaSX& a_ex_fun, const CaSX& a_le_fun) const {
    if (!constant_beta_expression_) {
      const auto beta_subst = CaSX::substitute(beta_, a_ex_, a_ex_fun);
      const auto beta_subst2 = CaSX::substitute(beta_subst, a_le_, a_le_fun);
      const auto beta_subst3 = CaSX::substitute(beta_subst2, x_, dm_->phi());
      return beta_subst3;
    }
    return CaSX::substitute(beta_, x_, dm_->phi());
  }

  CaSX substitute_eta() const { return eta_; }

  CaSXDict symbolic_parameters() const { return symbolic_params_; }

protected:
  CaSX beta_;
  CaSX eta_;
  CaSX x_;
  FabDifferentialMapPtr dm_ = nullptr;
  CaSX lg_;
  CaSX a_ex_;
  CaSX a_le_;
  CaSXDict symbolic_params_;
  bool constant_beta_expression_ = true;
};

class FabInterpolator {
public:
  FabInterpolator() = default;

  FabInterpolator(const CaSX& eta, FabLagrangianPtr lex, FabLagrangianPtr lex_d)
      : eta_(eta), lex_(std::move(lex)), lex_d_(std::move(lex_d)) {}

protected:
  CaSX eta_;
  FabLagrangianPtr lex_ = nullptr;
  FabLagrangianPtr lex_d_ = nullptr;
};