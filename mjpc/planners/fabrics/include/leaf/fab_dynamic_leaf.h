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

  FabDynamicLeaf(FabVariablesPtr parent_vars, std::string leaf_name, const int dim = 1, const int dim_ref = 1,
                 const CaSX& fk = CaSX::zeros(), CaSXDict reference_params = CaSXDict())
      : FabLeaf(std::move(parent_vars), std::move(leaf_name), fk, dim),
        dim_ref_(dim_ref),
        fk_x_(CASX_SYM(fk_x_, dim_ref)),
        fk_xdot_(CASX_SYM(fk_xdot_, dim_ref)),
        x_rel_(CASX_SYM(x_rel_, dim_ref)),
        xdot_rel_(CASX_SYM(xdot_rel_, dim_ref)),
        relative_vars_(std::make_shared<FabVariables>(
            CaSXDict{{LEAF_VAR_NAME(x_rel_), x_rel_}, {LEAF_VAR_NAME(xdot_rel_), xdot_rel_}})) {
    std::array<std::string, 3> ref_names;
    if (reference_params.empty()) {
      x_ref_ = CASX_SYM(x_ref_, dim_ref);
      xdot_ref_ = CASX_SYM(xdot_ref_, dim_ref);
      xddot_ref_ = CASX_SYM(xddot_ref_, dim_ref);

      ref_names[0] = LEAF_VAR_NAME(x_ref_);
      ref_names[1] = LEAF_VAR_NAME(xdot_ref_);
      ref_names[2] = LEAF_VAR_NAME(xddot_ref_);
      reference_params = {{ref_names[0], x_ref_}, {ref_names[1], xdot_ref_}, {ref_names[2], xddot_ref_}};
    } else {
      for (const auto& [param_name, param_val] : reference_params) {
        if (param_name.starts_with("x_")) {
          ref_names[0] = param_name;
          x_ref_ = param_val;
        } else if (param_name.starts_with("xdot_")) {
          ref_names[1] = param_name;
          xdot_ref_ = param_val;
        } else if (param_name.starts_with("xddot_")) {
          ref_names[2] = param_name;
          xddot_ref_ = param_val;
        }
      }
      assert(!x_ref_.is_empty());
      assert(!xdot_ref_.is_empty());
      assert(!xddot_ref_.is_empty());
    }

    fk_vars_ = std::make_shared<FabVariables>(
        CaSXDict{{LEAF_VAR_NAME(fk_x_), fk_x_}, {LEAF_VAR_NAME(fk_xdot_), fk_xdot_}}, reference_params);

#if 0
    const auto phi_dynamic = x_ - x_ref_;
    const auto phi_dot_dynamic = xdot_ - xdot_ref_;
    const auto Jdotqdot_dynamic = -xddot_ref_;
#endif
    dynamic_map_ = std::make_shared<FabDynamicDifferentialMap>(fk_vars_, ref_names);
    FAB_PRINTDB(dynamic_map_->ref_names());
  }

  FabDynamicDifferentialMapPtr dynamic_map() const { return dynamic_map_; }

  CaSX xdot_ref() const { return xdot_ref_; }

protected:
  int dim_ref_;
  CaSX fk_x_;
  CaSX fk_xdot_;
  CaSX x_rel_;
  CaSX xdot_rel_;

  FabVariablesPtr relative_vars_ = nullptr;
  FabVariablesPtr fk_vars_ = nullptr;
  CaSX x_ref_;
  CaSX xdot_ref_;
  CaSX xddot_ref_;
  FabDynamicDifferentialMapPtr dynamic_map_ = nullptr;
};
