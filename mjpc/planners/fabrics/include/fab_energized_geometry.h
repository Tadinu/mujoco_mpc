#pragma once

#include <casadi/casadi.hpp>
#include <cassert>
#include <memory>

#include "mjpc/planners/fabrics/include/fab_casadi_function.h"
#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_core_util.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_energy.h"
#include "mjpc/planners/fabrics/include/fab_geometry.h"
#include "mjpc/planners/fabrics/include/fab_spectral_semi_sprays.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"

class FabWeightedGeometry : public FabSpectralSemiSprays {
 public:
  FabWeightedGeometry() = default;

  explicit FabWeightedGeometry(const FabNamedMap<CaSX, FabVariables, FabGeometry, FabLagrangian,
                                                 FabSpectralSemiSprays, std::vector<std::string>>& kwargs) {
    // [x_ref_name_, xdot_ref_name_, xddot_ref_name_]
    if (kwargs.contains("ref_names")) {
      auto ref_names = *fab_core::get_arg_value<std::vector<std::string>>(kwargs, "ref_names");
      assert(ref_names.size() == 3);
      x_ref_name_ = std::move(ref_names[0]);
      xdot_ref_name_ = std::move(ref_names[1]);
      xddot_ref_name_ = std::move(ref_names[2]);
    }

    // [le_]
    if (kwargs.contains("le")) {
      le_ = *fab_core::get_arg_value<decltype(le_)>(kwargs, "le");
    }

    // [h_, M_, refTrajs_]
    if (kwargs.contains("g")) {
      const auto geom = *fab_core::get_arg_value<FabGeometry>(kwargs, "g");
      assert(fab_core::check_compatibility(le_, geom));
      vars_ = geom.vars() + le_.vars();
      refTrajs_ = FabVariables::join_refTrajs(le_.refTrajs(), geom.refTrajs());
      this->h_ = geom.h();
      this->M_ = le_.S().M();
    } else if (kwargs.contains("s")) {
      const auto s = *fab_core::get_arg_value<FabSpectralSemiSprays>(kwargs, "s");
      assert(fab_core::check_compatibility(le_, s));
      auto refTrajs = FabVariables::join_refTrajs(le_.refTrajs(), s.refTrajs());
      initialize(
          s.M(),
          {{"f", s.f()}, {"var", s.vars()}, {"refTrajs", std::move(refTrajs)}, {"ref_names", s.ref_names()}});
    }
  }

  CaSX x() const override { return le_.x(); }
  CaSX xdot() const override { return le_.xdot(); }
  CaSX alpha() const { return alpha_; }

  FabWeightedGeometry operator+(const FabWeightedGeometry& b) const {
    return FabWeightedGeometry(*this) += b;
  }

  FabWeightedGeometry& operator+=(const FabWeightedGeometry& b) {
    assert(fab_core::check_compatibility(*this, b));
    const auto spec = static_cast<FabSpectralSemiSprays>(*this) + static_cast<FabSpectralSemiSprays>(b);
    const auto all_le = le_ + b.le_;
    *this = FabWeightedGeometry({{"le", all_le}, {"s", spec}, {"ref_names", spec.ref_names()}});
    return *this;
  }

  void compute_alpha(int8_t ref_sign = 1) {
    const auto xdot = le_.xdot_rel(ref_sign);
    const auto frac = 1 / (FAB_EPS + CaSX::dot(xdot, CaSX::mtimes(le_.S().M(), xdot)));
    alpha_ = -frac * CaSX::dot(xdot, this->f() - le_.S().f());
  }

  void concretize(int8_t ref_sign = 1) {
    if (empty()) {
      return;
    }
    compute_alpha(ref_sign);
    xddot_ = -this->h();
    auto vars = vars_;
    for (const auto& refTraj : refTrajs_) {
      // TODO
      vars += refTraj;
    }

    func_ = std::make_shared<FabCasadiFunction>(
        "func_", vars, CaSXDict{{"M", this->M()}, {"f", this->f()}, {"xddot", xddot_}, {"alpha", alpha_}});
  }

  CaSXDict evaluate(const FabCasadiArgMap& kwargs) {
    if (func_) {
      auto eval = func_->evaluate(kwargs);
      return {{"M", eval["M"]}, {"f", eval["f"]}, {"xddot", eval["xddot"]}, {"alpha", eval["alpha"]}};
    }
    throw FabError::customized("FabWeightedGeometry evaluation failed", "Function not defined");
  }

  FabWeightedGeometry pull(const FabDifferentialMap& dm) const {
    auto spec = FabSpectralSemiSprays::pull(dm);
    auto le_pulled = le_.pull(dm);
    return FabWeightedGeometry(
        {{"s", std::move(spec)}, {"le", std::move(le_pulled)}, {"ref_names", ref_names()}});
  }

  FabWeightedGeometry dynamic_pull(const FabDynamicDifferentialMap& dm) const {
    auto spec = FabSpectralSemiSprays::dynamic_pull(dm);
    auto le_pulled = le_.dynamic_pull(dm);
    return FabWeightedGeometry(
        {{"s", std::move(spec)}, {"le", std::move(le_pulled)}, {"ref_names", dm.ref_names()}});
  }

 protected:
  FabLagrangian le_;
  CaSX alpha_;
};