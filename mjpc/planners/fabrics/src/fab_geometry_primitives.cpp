#include <casadi/casadi.hpp>

#include "mjpc/planners/fabrics/include/fab_geometry_primitives.h"

CaSX FabCapsule::distance(const FabGeometricPrimitive* primitive) {
  if (const auto* sphere_prim = dynamic_cast<const FabSphere*>(primitive)) {
    return fab_math::capsule_to_sphere(center(),
                                       sphere_prim->position(),
                                       sym_radius_,
                                       sphere_prim->sym_radius());
  }

  if (const auto* cuboid_prim = dynamic_cast<const FabCuboid*>(primitive)) {
    return fab_math::cuboid_to_capsule(cuboid_prim->position(),
                                       center(),
                                       cuboid_prim->size(),
                                       sym_radius_);
  }

  if (const auto* plane_prim = dynamic_cast<const FabPlane*>(primitive)) {
    return fab_math::capsule_to_plane(center(),
                                      plane_prim->sym_plane_equation(),
                                      sym_radius_);
  }
  throw FabDistanceUndefinedError::customized(this, primitive);
}

CaSX FabSphere::distance(const FabGeometricPrimitive* primitive) {
  if (const auto* sphere_prim = dynamic_cast<const FabSphere*>(primitive)) {
    return fab_math::sphere_to_sphere(position(),
                                      sphere_prim->position(),
                                      sym_radius_,
                                      sphere_prim->sym_radius());
  }

  if (const auto* plane_prim = dynamic_cast<const FabPlane*>(primitive)) {
    return fab_math::sphere_to_plane(position(),
                                     plane_prim->sym_plane_equation(),
                                     sym_radius_);
  }

  if (const auto* cuboid_prim = dynamic_cast<const FabCuboid*>(primitive)) {
    return fab_math::cuboid_to_sphere(cuboid_prim->position(),
                                      position(),
                                      cuboid_prim->size(),
                                      sym_radius_);
  }
  throw FabDistanceUndefinedError::customized(this, primitive);
}
