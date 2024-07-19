#pragma once
#include <casadi/casadi.hpp>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_energy.h"

class FabDamper {
 public:
  FabDamper() = default;

  FabDamper(const CaSX& x, const CaSX& beta, const CaSX& a_ex, const CaSX& a_le, const CaSX& eta,
            FabDifferentialMap dm, const CaSX& lagrangian_execution, bool const_beta_expression = false)
      : x_(x),
        beta_(beta),
        eta_(eta),
        dm_(std::move(dm)),
        lg_(lagrangian_execution),
        a_ex_(a_ex),
        a_le_(a_le),
        constant_beta_expression_(const_beta_expression) {}

  CaSX substitute_beta(const CaSX& a_ex_fun, const CaSX& a_le_fun) const {
    if (!constant_beta_expression_) {
      const auto beta_subst = CaSX::substitute(beta_, a_ex_, a_ex_fun);
      const auto beta_subst2 = CaSX::substitute(beta_subst, a_le_, a_le_fun);
      const auto beta_subst3 = CaSX::substitute(beta_subst2, x_, dm_.phi());
      return beta_subst3;
    }
    return CaSX::substitute(beta_, x_, dm_.phi());
  }

  CaSX substitute_eta() const { return eta_; }

  CaSXDict symbolic_parameters() const { return symbolic_params_; }

 protected:
  CaSX beta_;
  CaSX eta_;
  CaSX x_;
  FabDifferentialMap dm_;
  CaSX lg_;
  CaSX a_ex_;
  CaSX a_le_;
  CaSXDict symbolic_params_;
  bool constant_beta_expression_ = false;
};

class FabInterpolator {
 public:
  FabInterpolator() = default;

  FabInterpolator(CaSX eta, FabLagrangian lex, FabLagrangian lex_d)
      : eta_(std::move(eta)), lex_(std::move(lex)), lex_d_(std::move(lex_d)) {}

 protected:
  CaSX eta_;
  FabLagrangian lex_;
  FabLagrangian lex_d_;
};