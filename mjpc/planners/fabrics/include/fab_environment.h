#pragma once

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_geometry_primitives.h"

class FabEnvironment {
 public:
  FabEnvironment() = default;

  FabEnvironment(int spheres_num, int cuboids_num, int planes_num)
      : spheres_num_(spheres_num), cuboids_num_(cuboids_num), planes_num_(planes_num) {}

  void generate_obstacles() {
    size_t count = obstacles_.size();
    for (auto i = 0; i < spheres_num_; ++i) {
      auto obst_sphere =
          std::make_shared<FabSphere>(std::string("obst_sphere_") + std::to_string(count++), 1.0);
      obst_sphere->set_position(CaSX::sym(std::string("x_") + obst_sphere->name(), 3), true);
      obstacles_.push_back(std::move(obst_sphere));
    }

    for (auto i = 0; i < cuboids_num_; ++i) {
      auto obst_cuboid = std::make_shared<FabCuboid>(std::string("obst_cuboid_") + std::to_string(count++),
                                                     std::vector<double>{1.0, 1.0, 1.0});
      obst_cuboid->set_position(CaSX::sym(std::string("x_") + obst_cuboid->name(), 3), true);
      obstacles_.push_back(std::move(obst_cuboid));
    }

    for (auto i = 0; i < planes_num_; ++i) {
      auto constraint_plane =
          std::make_shared<FabPlane>(std::string("constraint_plane_") + std::to_string(i));
      constraint_plane->set_position(CaSX::sym(constraint_plane->name(), 3), true);
      obstacles_.push_back(std::move(constraint_plane));
    }
  }

  std::vector<FabGeometricPrimitivePtr> obstacles() const { return obstacles_; }

  int spheres_num() const { return spheres_num_; }
  int cuboids_num() const { return cuboids_num_; }
  int planes_num() const { return planes_num_; }

 protected:
  std::vector<FabGeometricPrimitivePtr> obstacles_;
  int spheres_num_ = 0;
  int cuboids_num_ = 0;
  int planes_num_ = 0;
};