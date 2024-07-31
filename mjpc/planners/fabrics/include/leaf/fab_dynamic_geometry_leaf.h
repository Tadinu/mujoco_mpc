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

  FabGenericDynamicGeometryLeaf(FabVariablesPtr parent_vars, std::string leaf_name, const int dim = 1,
                                const int dim_ref = 1, const CaSX& forward_kinematics = CaSX::zeros(),
                                const CaSXDict& reference_params = CaSXDict())
      : FabDynamicLeaf(std::move(parent_vars), std::move(leaf_name), dim, dim_ref, forward_kinematics,
                       reference_params) {}

  void set_geometry(const std::function<CaSX(const CaSX& x, const CaSX& xdot)>& geometry) {
    // TODO
    // new_parameters, h_geometry = parse_symbolic_input(geometry, x, xdot, name=self._leaf_name)
    // self._parent_variables.add_parameters(new_parameters)
    geom_ = std::make_shared<FabGeometry>(FabGeometryArgs{{"h", geometry(x_, xdot_)}, {"var", leaf_vars_}});
  }

  void set_finsler_structure(const std::function<CaSX(const CaSX& x, const CaSX& xdot)>& finsler_structure) {
    lag_ =
        std::make_shared<FabLagrangian>(finsler_structure(x_, xdot_), FabLagrangianArgs{{"var", leaf_vars_}});
  }

  virtual FabDifferentialMapPtr geometry_map() const { return nullptr; }
};

class FabDynamicObstacleLeaf : public FabGenericDynamicGeometryLeaf {
public:
  FabDynamicObstacleLeaf() = default;

  FabDynamicObstacleLeaf(FabVariablesPtr parent_vars, const CaSX& fk, const std::string& obstacle_name,
                         const std::string& collision_link_name,
                         const CaSXDict& reference_params = CaSXDict())
      : FabGenericDynamicGeometryLeaf(std::move(parent_vars),
                                      obstacle_name + "_" + collision_link_name + "_leaf", 1 /*dim*/,
                                      int(fk.size().first) /*dim_ref*/, fk, reference_params) {
    set_forward_map(obstacle_name, collision_link_name);
  }

  void set_forward_map(const std::string& obstacle_name, const std::string& collision_link_name) {
    FAB_PRINT("set_forward_map XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxx");
    parent_vars_->print_self();
    const auto radius_obstacle_name = std::string("radius_") + obstacle_name;
    const CaSX radius_obstacle_var = get_parent_var_param(radius_obstacle_name, 1);

    const auto radius_body_name = std::string("radius_body_") + collision_link_name;
    const CaSX radius_body_var = get_parent_var_param(radius_body_name, 1);

    geom_params_ = {{radius_obstacle_name, radius_obstacle_var}, {radius_body_name, radius_body_var}};
    parent_vars_->add_parameters(geom_params_);

    // Forward map
    diffmap_ = std::make_shared<FabDifferentialMap>(forward_kinematics_, parent_vars_);
    FAB_PRINT("DynamicObstacleLeaf set_forward_map =====================");
    print_self();

    // Geometry map
    geom_map_ =
        std::make_shared<FabSphereSphereMap>(relative_vars_, relative_vars_->position_var(),
                                             CaSX::zeros(dim_ref_), radius_obstacle_var, radius_body_var);
  }
  FabDifferentialMapPtr geometry_map() const override { return geom_map_; }

protected:
  FabDifferentialMapPtr geom_map_;
};