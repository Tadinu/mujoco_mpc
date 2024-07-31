#pragma once

#include <casadi/casadi.hpp>
#include <cassert>
#include <map>
#include <memory>

#include "mjpc/planners/fabrics/include/fab_casadi_function.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"

class FabGeometry {
public:
  FabGeometry() = default;
  virtual ~FabGeometry() = default;

  using FabGeometryPtr = std::shared_ptr<FabGeometry>;
  using FabGeometryArgs =
      FabNamedMap<CaSX, FabVariablesPtr, FabGeometryPtr, FabTrajectories, std::vector<std::string>>;

  explicit FabGeometry(const FabGeometryArgs& kwargs) {
    // [h_, vars_]
    if (kwargs.contains("x")) {
      assert(kwargs.contains("xdot"));
      h_ = *fab_core::get_arg_value<decltype(h_)>(kwargs, "h");
      vars_ =
          std::make_shared<FabVariables>(CaSXDict{{"x", *fab_core::get_arg_value<CaSX>(kwargs, "x")},
                                                  {"xdot", *fab_core::get_arg_value<CaSX>(kwargs, "xdot")}});
    } else if (kwargs.contains("var")) {
      h_ = *fab_core::get_arg_value<decltype(h_)>(kwargs, "h");
      vars_ = *fab_core::get_arg_value<decltype(vars_)>(kwargs, "var");
    } else if (kwargs.contains("s")) {
      const auto s = *fab_core::get_arg_value<FabGeometryPtr>(kwargs, "s");
      h_ = s->h();
      vars_ = s->vars();
    }

    // [refTrajs_]
    if (kwargs.contains("refTrajs")) {
      refTrajs_ = *fab_core::get_arg_value<decltype(refTrajs_)>(kwargs, "refTrajs");
    }
  }

  virtual CaSX x() const { return vars_->position_var(); }
  virtual CaSX xdot() const { return vars_->velocity_var(); }
  virtual CaSX xddot() const { return xddot_; }
  CaSX x_ref() const { return vars_->parameter(x_ref_name_); }
  CaSX xdot_ref() const { return vars_->parameter(xdot_ref_name_); }
  CaSX xddot_ref() const { return vars_->parameter(xddot_ref_name_); }

  FabVariablesPtr vars() const { return vars_; }

  std::vector<std::string> ref_names() const { return {x_ref_name_, xdot_ref_name_, xddot_ref_name_}; }
  // A dynamic geometry: one defined using relative coordinates
  bool is_dynamic() const { return vars_->parameters().contains(x_ref_name_); }

  FabTrajectories refTrajs() const { return refTrajs_; }
  FabTrajectory refTraj() const { return refTraj_; }

  virtual CaSX h() const { return h_; }
  bool has_h() const { return !h_.is_empty(); }

  FabGeometry operator+(const FabGeometry& b) const { return FabGeometry(*this) += b; }

  FabGeometry& operator+=(const FabGeometry& b) {
    assert(fab_core::check_compatibility(*this, b));
    h_ = h_ + b.h_;
    vars_ = std::make_shared<FabVariables>(*vars_ + *b.vars_);
    return *this;
  }

protected:
  virtual FabGeometryPtr pull(const FabDifferentialMap& dm) const {
    const auto h_pulled = CaSX::mtimes(CaSX::pinv(dm.J()), h_ + dm.Jdotqdot());
    const auto h_pulled_subst_x = CaSX::substitute(h_pulled, x(), dm.phi());
    const auto h_pulled_subst_x_xdot = CaSX::substitute(h_pulled_subst_x, xdot(), dm.phidot());

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
    return std::make_shared<FabGeometry>(
        FabGeometryArgs{{"h", h_pulled_subst_x_xdot}, {"var", std::move(new_vars)}, {"refTrajs", refTrajs}});
  }

  virtual FabGeometryPtr dynamic_pull(const FabDynamicDifferentialMap& dm) const {
    const auto h_pulled = h_ - dm.xddot_ref();
    const auto h_pulled_subst_x = CaSX::substitute(h_pulled, x(), dm.phi());
    const auto h_pulled_subst_x_xdot = CaSX::substitute(h_pulled_subst_x, xdot(), dm.phidot());
    return std::make_shared<FabGeometry>(FabGeometryArgs{{"h", h_pulled_subst_x_xdot}, {"var", dm.vars()}});
  }

public:
  template <typename TGeometry, typename TGeometryPtr = std::shared_ptr<TGeometry>,
            typename = std::enable_if<!std::is_same_v<TGeometry, FabGeometry> &&
                                      std::is_base_of_v<TGeometry, FabGeometry>>>
  TGeometryPtr pull_back(const FabDifferentialMap& dm) const {
    return std::dynamic_pointer_cast<TGeometry>(pull(dm));
  }

  template <typename TGeometry, typename TGeometryPtr = std::shared_ptr<TGeometry>,
            typename = std::enable_if<!std::is_same_v<TGeometry, FabGeometry> &&
                                      std::is_base_of_v<TGeometry, FabGeometry>>>
  TGeometryPtr dynamic_pull_back(const FabDynamicDifferentialMap& dm) const {
    return std::dynamic_pointer_cast<TGeometry>(dynamic_pull(dm));
  }

  virtual void concretize(int8_t ref_sign = 1) {
    xddot_ = -h_;
    auto vars = *vars_;
    for (const auto& refTraj : refTrajs_) {
      // TODO
      vars += refTraj;
    }

    func_ =
        std::make_shared<FabCasadiFunction>("func_", std::move(vars), CaSXDict{{"h", h_}, {"xddot", xddot_}});
  }

  virtual CaSXDict evaluate(const FabCasadiArgMap& kwargs) const {
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
  FabVariablesPtr vars_ = nullptr;
  CaSX h_;
  CaSX xddot_;
  std::shared_ptr<FabCasadiFunction> func_ = nullptr;
  FabTrajectories refTrajs_;
  FabTrajectory refTraj_;
};

using FabGeometryPtr = FabGeometry::FabGeometryPtr;
using FabGeometryArgs = FabGeometry::FabGeometryArgs;
