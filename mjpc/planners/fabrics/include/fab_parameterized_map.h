#pragma once

#include <casadi/casadi.hpp>
#include <memory>

#include "mjpc/planners/fabrics/include/fab_casadi_function.h"
#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_core_util.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_math_util.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"

class FabParameterizedGoalMap : public FabDifferentialMap {
 public:
  FabParameterizedGoalMap() = default;

  FabParameterizedGoalMap(FabVariables vars, const CaSX& fk, const CaSX& reference_var)
      : FabDifferentialMap(fk - reference_var, std::move(vars)) {}
};

class FabParameterizedGeometryMap : public FabDifferentialMap {
 public:
  FabParameterizedGeometryMap() = default;

  FabParameterizedGeometryMap(CaSX phi, FabVariables vars)
      : FabDifferentialMap(std::move(phi), std::move(vars)) {}
};

class FabParameterizedObstacleMap : public FabParameterizedGeometryMap {
 public:
  FabParameterizedObstacleMap() = default;

  FabParameterizedObstacleMap(FabVariables vars, const CaSX& fk, const CaSX& reference_var,
                              const CaSX& radius_var, const CaSX& radius_body_var)
      : FabParameterizedGeometryMap(CaSX::norm_2(fk - reference_var) / (radius_var + radius_body_var) - 1,
                                    std::move(vars)) {}
};

class FabSphereSphereMap : public FabParameterizedGeometryMap {
 public:
  FabSphereSphereMap(FabVariables vars, const CaSX& sphere_1_position, const CaSX& sphere_2_position,
                     const CaSX& sphere_1_radius, const CaSX& sphere_2_radius)
      : FabParameterizedGeometryMap(
            CaSX::norm_2(sphere_1_position - sphere_2_position) / (sphere_1_radius + sphere_2_radius) - 1,
            std::move(vars)) {}
};

class FabCapsuleSphereMap : public FabParameterizedGeometryMap {
 public:
  FabCapsuleSphereMap(FabVariables vars, const CaSXVector& capsule_centers, const CaSX& sphere_center,
                      const CaSX& capsule_radius, const CaSX& sphere_radius)
      : FabParameterizedGeometryMap(
            fab_math::capsule_to_sphere(capsule_centers, sphere_center, capsule_radius, sphere_radius),
            std::move(vars)) {}
};

class FabCapsuleCuboidMap : public FabParameterizedGeometryMap {
 public:
  FabCapsuleCuboidMap(FabVariables vars, const CaSXVector& capsule_centers, const CaSX& cuboid_center,
                      const CaSX& capsule_radius, const CaSX& cuboid_size)
      : FabParameterizedGeometryMap(
            fab_math::cuboid_to_capsule(cuboid_center, capsule_centers, cuboid_size, capsule_radius),
            std::move(vars)) {}
};

class FabPlaneSphereMap : public FabParameterizedGeometryMap {
 public:
  FabPlaneSphereMap(FabVariables vars, const CaSX& sphere_center, const CaSX& sphere_radius,
                    const CaSX& constraint)
      : FabParameterizedGeometryMap(fab_math::sphere_to_plane(sphere_center, constraint, sphere_radius),
                                    std::move(vars)) {}
};

class FabCuboidSphereMap : public FabParameterizedGeometryMap {
 public:
  FabCuboidSphereMap(FabVariables vars, const CaSX& sphere_center, const CaSX& cuboid_center,
                     const CaSX& sphere_radius, const CaSX& cuboid_size)
      : FabParameterizedGeometryMap(
            fab_math::cuboid_to_sphere(cuboid_center, sphere_center, cuboid_size, sphere_radius),
            std::move(vars)) {}
};

class FabParameterizedPlaneConstraintMap : public FabParameterizedGeometryMap {
 public:
  FabParameterizedPlaneConstraintMap(FabVariables vars, const CaSX& fk, const CaSX& constraint_var,
                                     const CaSX& radius_body_var)
      : FabParameterizedGeometryMap(CaSX::abs(CaSX::dot(fab_core::get_casx(constraint_var, 0, 3), fk) +
                                              fab_core::get_casx(constraint_var, 3)) /
                                            CaSX::norm_2(fab_core::get_casx(constraint_var, 0, 3)) -
                                        radius_body_var,
                                    std::move(vars)) {}
};
