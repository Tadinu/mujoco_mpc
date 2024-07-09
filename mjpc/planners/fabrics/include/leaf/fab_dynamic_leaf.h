#pragma once

#include <casadi/casadi.hpp>
#include <map>
#include <memory>
#include <stdexcept>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_energy.h"
#include "mjpc/planners/fabrics/include/fab_geometry.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"

class FabDynamicLeaf : public FabLeaf {
 public:
  FabDynamicLeaf() = default;

  FabDynamicLeaf(FabVariables parent_variables, std::string leaf_name, const int dim = 1,
                 const int dim_ref = 1, CaSX fk = CaSX::zeros(),
                 const CaSXDict& reference_params = CaSXDict())
      : FabLeaf(std::move(parent_variables), std::move(leaf_name), std::move(fk), dim),
        dim_ref_(dim_ref),
        fk_x_(CaSX::sym(std::string("fk_x_") + leaf_name_, dim_ref)),
        fk_xdot_(CaSX::sym(std::string("fk_xdot_") + leaf_name_, dim_ref)),
        xrel_(CaSX::sym(std::string("x_rel_") + leaf_name_, dim_ref)),
        xdot_rel_(CaSX::sym(std::string("xdot_rel_") + leaf_name_, dim_ref)),
        relative_vars_(FabVariables({{xrel_.name(), xrel_}, {xdot_rel_.name(), xdot_rel_}})) {
    CaSXDict ref_params;
    if (reference_params.empty()) {
      x_ref_ = CaSX::sym(std::string("x_ref_") + leaf_name_, dim_ref);
      xdot_ref_ = CaSX::sym(std::string("xdot_ref_") + leaf_name_, dim_ref);
      xddot_ref_ = CaSX::sym(std::string("xddot_ref_") + leaf_name_, dim_ref);
      ref_params = {{x_ref_.name(), x_ref_}, {xdot_ref_.name(), xdot_ref_}, {xddot_ref_.name(), xddot_ref_}};
    } else {
      ref_params = reference_params;
      const auto ref_params_values = fab_core::get_map_values<CaSX>(reference_params);
      x_ref_ = ref_params_values[0];
      xdot_ref_ = ref_params_values[1];
      xddot_ref_ = ref_params_values[2];
    }

    fk_vars_ = FabVariables({{"fk_x_{leaf_name}", fk_x_}, {"fk_xdot_{leaf_name}", fk_xdot_}}, ref_params);
    diffmap_ = std::make_shared<FabDynamicDifferentialMap>(fk_vars_, fab_core::get_map_keys(ref_params));

#if 0
    const auto phi_dynamic = x_ - x_ref_;
    const auto phi_dot_dynamic = xdot_ - xdot_ref_;
    const auto Jdotqdot_dynamic = -xddot_ref_;
#endif
  }

  void set_params(const CaSXDict& kwargs) {
    for (const auto& [key, _] : p_) {
      if (kwargs.contains(key)) {
        p_.insert_or_assign(key, kwargs.at(key));
      }
    }
  }

  std::shared_ptr<FabDynamicDifferentialMap> dynamic_map() const {
    return std::dynamic_pointer_cast<FabDynamicDifferentialMap>(diffmap_);
  }

  CaSX xdot_ref() const { return xdot_ref_; }

 protected:
  int dim_ref_;
  CaSX fk_x_;
  CaSX fk_xdot_;
  CaSX xrel_;
  CaSX xdot_rel_;

  FabVariables relative_vars_;
  FabVariables fk_vars_;
  CaSX x_ref_;
  CaSX xdot_ref_;
  CaSX xddot_ref_;
};
