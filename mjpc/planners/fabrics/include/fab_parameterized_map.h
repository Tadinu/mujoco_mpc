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

  FabParameterizedGoalMap(std::string name, const FabVariablesPtr& vars, const CaSX& fk,
                          const CaSX& reference_var)
      : FabDifferentialMap(std::move(name), fk - reference_var, vars) {}
};

class FabParameterizedGeometryMap : public FabDifferentialMap {
public:
  FabParameterizedGeometryMap() = default;

  FabParameterizedGeometryMap(std::string name, const CaSX& phi, const FabVariablesPtr& vars)
      : FabDifferentialMap(std::move(name), phi, vars) {}
};

class FabParameterizedObstacleMap : public FabParameterizedGeometryMap {
public:
  FabParameterizedObstacleMap() = default;

  FabParameterizedObstacleMap(std::string name, const FabVariablesPtr& vars, const CaSX& fk,
                              const CaSX& reference_var, const CaSX& radius_var, const CaSX& radius_body_var)
      : FabParameterizedGeometryMap(
            std::move(name), CaSX::norm_2(fk - reference_var) / (radius_var + radius_body_var) - 1, vars) {}
};

class FabSphereSphereMap : public FabParameterizedGeometryMap {
public:
  FabSphereSphereMap(std::string name, const FabVariablesPtr& vars, const CaSX& sphere_1_position,
                     const CaSX& sphere_2_position, const CaSX& sphere_1_radius, const CaSX& sphere_2_radius)
      : FabParameterizedGeometryMap(
            std::move(name),
            CaSX::norm_2(sphere_1_position - sphere_2_position) / (sphere_1_radius + sphere_2_radius) - 1,
            vars) {}
};

class FabCapsuleSphereMap : public FabParameterizedGeometryMap {
public:
  FabCapsuleSphereMap(std::string name, const FabVariablesPtr& vars, const CaSXVector& capsule_centers,
                      const CaSX& sphere_center, const CaSX& capsule_radius, const CaSX& sphere_radius)
      : FabParameterizedGeometryMap(
            std::move(name),
            fab_math::capsule_to_sphere(capsule_centers, sphere_center, capsule_radius, sphere_radius),
            vars) {}
};

class FabCapsuleCuboidMap : public FabParameterizedGeometryMap {
public:
  FabCapsuleCuboidMap(std::string name, const FabVariablesPtr& vars, const CaSXVector& capsule_centers,
                      const CaSX& cuboid_center, const CaSX& capsule_radius, const CaSX& cuboid_size)
      : FabParameterizedGeometryMap(
            std::move(name),
            fab_math::cuboid_to_capsule(cuboid_center, capsule_centers, cuboid_size, capsule_radius), vars) {}
};

class FabPlaneSphereMap : public FabParameterizedGeometryMap {
public:
  FabPlaneSphereMap(std::string name, const FabVariablesPtr& vars, const CaSX& sphere_center,
                    const CaSX& sphere_radius, const CaSX& constraint)
      : FabParameterizedGeometryMap(
            std::move(name), fab_math::sphere_to_plane(sphere_center, constraint, sphere_radius), vars) {}
};

class FabCuboidSphereMap : public FabParameterizedGeometryMap {
public:
  FabCuboidSphereMap(std::string name, const FabVariablesPtr& vars, const CaSX& sphere_center,
                     const CaSX& cuboid_center, const CaSX& sphere_radius, const CaSX& cuboid_size)
      : FabParameterizedGeometryMap(
            std::move(name),
            fab_math::cuboid_to_sphere(cuboid_center, sphere_center, cuboid_size, sphere_radius), vars) {}
};

class FabParameterizedPlaneConstraintMap : public FabParameterizedGeometryMap {
public:
  FabParameterizedPlaneConstraintMap(std::string name, const FabVariablesPtr& vars, const CaSX& fk,
                                     const CaSX& constraint_var, const CaSX& radius_body_var)
      : FabParameterizedGeometryMap(
            std::move(name),
            CaSX::abs(CaSX::dot(fab_core::get_casx(constraint_var, std::array<casadi_int, 2>{0, 3}), fk) +
                      fab_core::get_casx(constraint_var, 3)) /
                    CaSX::norm_2(fab_core::get_casx(constraint_var, std::array<casadi_int, 2>{0, 3})) -
                radius_body_var,
            vars) {}
};
