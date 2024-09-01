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
 * The geometry's geometry and metric are defined through the corresponding symbolic expression functions
 *
 * In contrast to FabGenericGeometryLeaf, FabGenericDynamicGeometryLeaf has an additional differential map,
 * namely a RelativeDifferentialMap.
 */
class FabGenericDynamicGeometryLeaf : public FabDynamicLeaf {
public:
  FabGenericDynamicGeometryLeaf() = default;

  FabGenericDynamicGeometryLeaf(std::string leaf_name, FabVariablesPtr parent_vars, const int dim = 1,
                                const int dim_ref = 1, const CaSX& forward_kinematics = CaSX::zeros(),
                                const CaSXDict& reference_params = CaSXDict())
      : FabDynamicLeaf(std::move(leaf_name), std::move(parent_vars), dim, dim_ref, forward_kinematics,
                       reference_params) {}

  void set_geometry(const FabConfigFunc& geometry) {
    const auto [h_geometry, var_names] = geometry(x_, xdot_, leaf_name_);
    parent_vars_->add_parameters(fab_core::parse_symbolic_casx(h_geometry, var_names));
    geom_ = std::make_shared<FabGeometry>(name() + "_geom",
                                          FabGeometryArgs{{"h", h_geometry}, {"var", leaf_vars_}});
  }

  void set_finsler_structure(const FabConfigFunc& finsler_structure) {
    const auto [lag_geometry, var_names] = finsler_structure(x_, xdot_, leaf_name_);
    parent_vars_->add_parameters(fab_core::parse_symbolic_casx(lag_geometry, var_names));
    lag_ = std::make_shared<FabLagrangian>(name() + "_lag", lag_geometry,
                                           FabLagrangianArgs{{"var", leaf_vars_}});
  }

  virtual FabDifferentialMapPtr geometry_map() const { return nullptr; }
};

class FabDynamicObstacleLeaf : public FabGenericDynamicGeometryLeaf {
public:
  FabDynamicObstacleLeaf() = default;

  FabDynamicObstacleLeaf(FabVariablesPtr parent_vars, const CaSX& fk, const std::string& obstacle_name,
                         const std::string& collision_link_name,
                         const CaSXDict& reference_params = CaSXDict())
      : FabGenericDynamicGeometryLeaf(obstacle_name + "_" + collision_link_name + "_leaf",
                                      std::move(parent_vars), 1 /*dim*/, int(fk.size().first) /*dim_ref*/, fk,
                                      reference_params) {
    set_forward_map(obstacle_name, collision_link_name);
  }

  FabDifferentialMapPtr geometry_map() const override { return geom_map_; }

private:
  void set_forward_map(const std::string& obstacle_name, const std::string& collision_link_name) {
    const auto radius_obstacle_name = "radius_" + obstacle_name;
    const CaSX radius_obstacle_var = get_parent_var_param(radius_obstacle_name, 1);

    const auto radius_body_name = "radius_body_" + collision_link_name;
    const CaSX radius_body_var = get_parent_var_param(radius_body_name, 1);

    geom_params_ = {{radius_obstacle_name, radius_obstacle_var}, {radius_body_name, radius_body_var}};
    parent_vars_->add_parameters(geom_params_);

    // Forward map
    diffmap_ = std::make_shared<FabDifferentialMap>(name() + "_diffmap", forward_kinematics_, parent_vars_);

    // Geometry map
    geom_map_ = std::make_shared<FabSphereSphereMap>(name() + "_geom_map", relative_vars_,
                                                     relative_vars_->position_var(), CaSX::zeros(dim_ref_),
                                                     radius_obstacle_var, radius_body_var);
  }

protected:
  FabDifferentialMapPtr geom_map_;
};