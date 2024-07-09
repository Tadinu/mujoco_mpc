#pragma once
#include <casadi/casadi.hpp>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_energy.h"

class FabDamper {
 public:
  FabDamper() = default;

  FabDamper(CaSX x, CaSX beta, CaSX a_ex, CaSX a_le, CaSX eta, FabDifferentialMap dm,
            CaSX lagrangian_execution, bool const_beta_expression = false)
      : x_(std::move(x)),
        beta_(std::move(beta)),
        eta_(std::move(eta)),
        dm_(std::move(dm)),
        lg_(std::move(lagrangian_execution)),
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
  bool constant_beta_expression_ = false;
  CaSX beta_;
  CaSX eta_;
  CaSX x_;
  FabDifferentialMap dm_;
  CaSX lg_;
  CaSX a_ex_;
  CaSX a_le_;
  CaSXDict symbolic_params_;
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