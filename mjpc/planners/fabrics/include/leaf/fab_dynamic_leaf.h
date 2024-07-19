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
                 const int dim_ref = 1, const CaSX& fk = CaSX::zeros(),
                 CaSXDict reference_params = CaSXDict())
      : FabLeaf(std::move(parent_variables), std::move(leaf_name), fk, dim),
        dim_ref_(dim_ref),
        fk_x_(CASX_SYM(fk_x_, dim_ref)),
        fk_xdot_(CASX_SYM(fk_xdot_, dim_ref)),
        x_rel_(CASX_SYM(x_rel_, dim_ref)),
        xdot_rel_(CASX_SYM(xdot_rel_, dim_ref)),
        relative_vars_(
            FabVariables({{LEAF_VAR_NAME(x_rel_), x_rel_}, {LEAF_VAR_NAME(xdot_rel_), xdot_rel_}})) {
    CaSXDict ref_params;
    if (reference_params.empty()) {
      x_ref_ = CASX_SYM(x_ref_, dim_ref);
      xdot_ref_ = CASX_SYM(xdot_ref_, dim_ref);
      xddot_ref_ = CASX_SYM(xddot_ref_, dim_ref);
      ref_params = {{LEAF_VAR_NAME(x_ref_), x_ref_},
                    {LEAF_VAR_NAME(xdot_ref_), xdot_ref_},
                    {LEAF_VAR_NAME(xddot_ref_), xddot_ref_}};
    } else {
      ref_params = std::move(reference_params);
      auto ref_params_values = fab_core::get_map_values<CaSX>(ref_params);
      assert(ref_params_values.size() == 3);
      x_ref_ = ref_params_values[0];
      xdot_ref_ = ref_params_values[1];
      xddot_ref_ = ref_params_values[2];
    }

    fk_vars_ = FabVariables({{LEAF_VAR_NAME(fk_x_), fk_x_}, {LEAF_VAR_NAME(fk_xdot_), fk_xdot_}}, ref_params);
    diffmap_ = std::make_shared<FabDynamicDifferentialMap>(fk_vars_, fab_core::get_map_keys(ref_params));

#if 0
    const auto phi_dynamic = x_ - x_ref_;
    const auto phi_dot_dynamic = xdot_ - xdot_ref_;
    const auto Jdotqdot_dynamic = -xddot_ref_;
#endif
  }

  std::shared_ptr<FabDynamicDifferentialMap> dynamic_map() const {
    return std::dynamic_pointer_cast<FabDynamicDifferentialMap>(diffmap_);
  }

  CaSX xdot_ref() const { return xdot_ref_; }

 protected:
  int dim_ref_;
  CaSX fk_x_;
  CaSX fk_xdot_;
  CaSX x_rel_;
  CaSX xdot_rel_;

  FabVariables relative_vars_;
  FabVariables fk_vars_;
  CaSX x_ref_;
  CaSX xdot_ref_;
  CaSX xddot_ref_;
};
