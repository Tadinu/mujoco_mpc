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
#include "mjpc/planners/fabrics/include/leaf/fab_dynamic_leaf.h"

/*
 * The FabGenericDynamicGeometryLeaf is a leaf to the tree of fabrics.
 * The geometry's geometry and metric are defined through the corresponding functions
 * to which the symbolic expression is passed as a string.
 *
 * In contrast to the GenericGeometry, the GenericDynamicGeometry has an additional differential map,
 * namely a RelativeDifferentialMap.
 */
class FabGenericDynamicGeometryLeaf : public FabDynamicLeaf {
 public:
  FabGenericDynamicGeometryLeaf() = default;
  FabGenericDynamicGeometryLeaf(FabVariables parent_variables, std::string leaf_name, const int dim = 1,
                                const int dim_ref = 1, CaSX forward_kinematics = CaSX::zeros(),
                                const CaSXDict& reference_params = CaSXDict())
      : FabDynamicLeaf(std::move(parent_variables), std::move(leaf_name), dim, dim_ref,
                       std::move(forward_kinematics), reference_params) {}

  void set_geometry(const std::function<CaSX(const CaSX& x, const CaSX& xdot)>& geometry) {
    geom_ = FabWeightedGeometry({{"h", geometry(x_, xdot_)}, {"var", leaf_vars_}});
  }

  void set_finsler_structure(const std::function<CaSX(const CaSX& x, const CaSX& xdot)>& finsler_structure) {
    lag_ = FabLagrangian(finsler_structure(x_, xdot_), {{"var", leaf_vars_}});
  }
};

class FabDynamicObstacleLeaf : public FabGenericDynamicGeometryLeaf {
 public:
  FabDynamicObstacleLeaf() = default;

  FabDynamicObstacleLeaf(FabVariables parent_vars, const CaSX& fk, const std::string& obstacle_name,
                         const std::string& collision_link_name,
                         const CaSXDict& reference_params = CaSXDict())
      : FabGenericDynamicGeometryLeaf(std::move(parent_vars),
                                      obstacle_name + "_" + collision_link_name + "_leaf", 1, fk.size().first,
                                      fk, reference_params) {
    set_forward_map(obstacle_name, collision_link_name);
  }

  void set_forward_map(const std::string& obstacle_name, const std::string& collision_link_name) {}
};