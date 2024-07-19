#pragma once

#include <casadi/casadi.hpp>
#include <map>
#include <memory>
#include <stdexcept>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_energy.h"
#include "mjpc/planners/fabrics/include/fab_geometry.h"
#include "mjpc/planners/fabrics/include/fab_parameterized_map.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"
#include "mjpc/planners/fabrics/include/leaf/fab_leaf.h"

class FabGenericGeometryLeaf : public FabLeaf {
public:
  FabGenericGeometryLeaf() = default;

  FabGenericGeometryLeaf(FabVariablesPtr parent_vars, std::string leaf_name, const CaSX& phi = CaSX())
      : FabLeaf(std::move(parent_vars), std::move(leaf_name), phi) {}

  void set_geometry(const std::function<CaSX(const CaSX& x, const CaSX& xdot)>& geometry) {
    // TODO
    // new_parameters, h_geometry = parse_symbolic_input(geometry, x, xdot, name=self._leaf_name)
    // self._parent_variables.add_parameters(new_parameters)
    geom_ = std::make_shared<FabGeometry>(FabGeometryArgs{{"h", geometry(x_, xdot_)}, {"var", leaf_vars_}});
  }

  void set_finsler_structure(const std::function<CaSX(const CaSX& x, const CaSX& xdot)>& finsler_structure) {
    // TODO
    // new_parameters, lagrangian_geometry = parse_symbolic_input(finsler_structure, x_, xdot_,
    // name=self._leaf_name) self._parent_variables.add_parameters(new_parameters)
    lag_ =
        std::make_shared<FabLagrangian>(finsler_structure(x_, xdot_), FabLagrangianArgs{{"var", leaf_vars_}});
  }
};

class FabAvoidanceLeaf : public FabGenericGeometryLeaf {
public:
  FabAvoidanceLeaf(FabVariablesPtr parent_vars, std::string leaf_name, const CaSX& phi = CaSX())
      : FabGenericGeometryLeaf(std::move(parent_vars), std::move(leaf_name), phi) {}
};

class FabLimitLeaf : public FabGenericGeometryLeaf {
public:
  FabLimitLeaf(const FabVariablesPtr& parent_vars, const int joint_index, const double limit,
               const int limit_index)
      : FabGenericGeometryLeaf(
            parent_vars,
            (std::string("limit_joint_") + std::to_string(joint_index) + "_" + std::to_string(limit_index)) +
                "_leaf",
            ((limit_index == 0)   ? (fab_core::get_casx(parent_vars->position_var(), joint_index) - limit)
             : (limit_index == 1) ? (limit - fab_core::get_casx(parent_vars->position_var(), joint_index))
                                  : CaSX())) {
    set_forward_map();
  }

private:
  void set_forward_map() {
    diffmap_ = std::make_shared<FabDifferentialMap>(forward_kinematics_, parent_vars_);
  }
};

/*
 * FabSelfCollisionLeaf is a geometry leaf for self collision avoidanceself.
 * This leaf is not parameterized as it is not changing at runtimeself.
 */
class FabSelfCollisionLeaf : public FabGenericGeometryLeaf {
public:
  FabSelfCollisionLeaf() = default;

  FabSelfCollisionLeaf(FabVariablesPtr parent_vars, const CaSX& fk, const std::string& collision_link_1_name,
                       const std::string& collision_link_2_name)
      : FabGenericGeometryLeaf(
            parent_vars, std::string("self_collision_") + collision_link_1_name + "_" + collision_link_2_name,
            fk) {
    set_forward_map(collision_link_1_name, collision_link_2_name);
  }

private:
  void set_forward_map(const std::string& collision_link_1_name, const std::string& collision_link_2_name) {
    const auto radius_body_1_name = std::string("radius_body_") + collision_link_1_name;
    const auto radius_body_2_name = std::string("radius_body_") + collision_link_2_name;
    const CaSX radius_body_1_var = get_parent_var_param(radius_body_1_name, 1);
    const CaSX radius_body_2_var = get_parent_var_param(radius_body_2_name, 1);
    parent_vars_->add_parameters(
        {{radius_body_1_name, radius_body_1_var}, {radius_body_2_name, radius_body_2_var}});
    auto phi = CaSX::norm_2(forward_kinematics_) / (radius_body_1_var + radius_body_2_var) - 1;
    diffmap_ = std::make_shared<FabDifferentialMap>(std::move(phi), parent_vars_);
  }
};

/*
 * FabObstacleLeaf is a geometry leaf for spherical obstacles.
 * The obstacles are parameterized by the obstacles radius, its position and the radius of the encapsulating
 * sphere for the corresponding link. Moreover, the symbolic expression for the forward expression is passed
 * to the constructor.
 */
class FabObstacleLeaf : public FabGenericGeometryLeaf {
public:
  FabObstacleLeaf() = default;

  FabObstacleLeaf(FabVariablesPtr parent_vars, const CaSX& fk, const std::string& obstacle_name,
                  const std::string& collision_link_name)
      : FabGenericGeometryLeaf(std::move(parent_vars), obstacle_name + "_" + collision_link_name + "_leaf",
                               fk) {
    set_forward_map(obstacle_name, collision_link_name);
  }

private:
  void set_forward_map(const std::string& obstacle_name, const std::string& collision_link_name) {
    const auto radius_obstacle_name = std::string("radius_") + obstacle_name;
    const CaSX radius_obstacle_var = get_parent_var_param(radius_obstacle_name, 1);

    const auto radius_body_name = std::string("radius_body_") + collision_link_name;
    const CaSX radius_body_var = get_parent_var_param(radius_body_name, 1);

    const auto obstacle_dim = forward_kinematics_.size().first;
    const auto reference_name = std::string("x_") + obstacle_name;
    const CaSX reference_var = get_parent_var_param(reference_name, obstacle_dim);

    geom_params_ = {{reference_name, reference_var},
                    {radius_obstacle_name, radius_obstacle_var},
                    {radius_body_name, radius_body_var}};
    parent_vars_->add_parameters(geom_params_);

#if 1
    const auto& sphere_1_pos_var = reference_var;
    const auto& sphere_2_pos_var = forward_kinematics_;
    diffmap_ = std::make_shared<FabSphereSphereMap>(parent_vars_, sphere_1_pos_var, sphere_2_pos_var,
                                                    radius_body_var, radius_obstacle_var);
#else
    diffmap_ = std::make_shared<FabParameterizedObstacleMap>(parent_vars_, forward_kinematics_, reference_var,
                                                             radius_body_var, radius_var);
#endif
  }
};

/*
 * ESDFGeometryLeaf is a leaf with explicit gradients that can be set at runtime.
 * Euclidean Signed Distance Fields (ESDF) can be exploited to avoid explicit geometry representations.
 * As automated differentiation does not work on ESDFs, this GeometryLeaf adds parameters for J and Jdot
 * that can be computed at runtime.
 *
 * Note that the signed distance to the closest obstacle is given by phi which is a function of a Euclidean
 * position which depends on the forward kinematics of the robot, phi(fk(q)). When computed the gradient,
 * one has to respect the chain rule leading to the jacobian of phi as:
 * d phi / d q = d phi / d x * d x / d q.
 *
 * The second part is the gradient of the forward kinematics which can be auto generated,
 * see J_collision_link in set_forward_map.
 * At runtime, one has to specify phi and d phi / d x.
 */
class FabESDFGeometryLeaf : public FabGenericGeometryLeaf {
public:
  FabESDFGeometryLeaf() = default;

  FabESDFGeometryLeaf(FabVariablesPtr parent_vars, std::string collision_link_name, const CaSX& collision_fk)
      : FabGenericGeometryLeaf(std::move(parent_vars), std::string("esdf_leaf_") + collision_link_name,
                               CaSX::sym(std::string("esdf_phi_") + collision_link_name, 1)),
        collision_link_name_(std::move(collision_link_name)),
        collision_fk_(collision_fk) {
    set_forward_map();
  }

private:
  void set_forward_map() {
    const CaSX q = parent_vars_->position_var();
    const CaSX J_collision_link = CaSX::jacobian(collision_fk_, q);
    const CaSX J_esdf = CaSX::sym(std::string("esdf_J_") + collision_link_name_, 3).T();

    CaSX J = CaSX::mtimes(J_esdf, J_collision_link);
    CaSX Jdot_esdf = CaSX::sym(std::string("esdf_Jdot_") + collision_link_name_, q.size().first).T();
    const auto radius_body_name = std::string("radius_body_") + collision_link_name_;
    const CaSXDict explicit_jacobians = {
        {std::string("esdf_phi_") + collision_link_name_, forward_kinematics_},
        {std::string("esdf_J_") + collision_link_name_, J_esdf},
        {std::string("esdf_Jdot_") + collision_link_name_, Jdot_esdf}};
    parent_vars_->add_parameters(explicit_jacobians);

    const CaSX radius_body_var = get_parent_var_param(radius_body_name, 1);
    parent_vars_->add_parameters({{radius_body_name, radius_body_var}});

    CaSX phi_reduced = forward_kinematics_ - radius_body_var;
    diffmap_ = std::make_shared<FabExplicitDifferentialMap>(
        std::move(phi_reduced), parent_vars_,
        FabDiffMapArg{{"J", std::move(J)}, {"Jdot", std::move(Jdot_esdf)}});
  }

protected:
  std::string collision_link_name_;
  CaSX collision_fk_;
};

class FabPlaneConstraintGeometryLeaf : public FabGenericGeometryLeaf {
public:
  FabPlaneConstraintGeometryLeaf() = default;

  FabPlaneConstraintGeometryLeaf(FabVariablesPtr parent_vars, std::string constraint_name,
                                 std::string collision_link_name, const CaSX& collision_fk)
      : FabGenericGeometryLeaf(std::move(parent_vars), collision_link_name + "_" + constraint_name,
                               collision_fk),
        constraint_name_(std::move(constraint_name)),
        collision_link_name_(std::move(collision_link_name)),
        collision_fk_(collision_fk) {
    set_forward_map();
  }

private:
  void set_forward_map() {
    const CaSX q = parent_vars_->position_var();
    const auto radius_body_name = std::string("radius_body_") + collision_link_name_;
    const CaSX radius_body_var = get_parent_var_param(radius_body_name, 1);
    const CaSX constraint_var = get_parent_var_param(constraint_name_, 4);

    parent_vars_->add_parameters({{radius_body_name, radius_body_var}, {constraint_name_, constraint_var}});
    diffmap_ = std::make_shared<FabParameterizedPlaneConstraintMap>(parent_vars_, forward_kinematics_,
                                                                    constraint_var, radius_body_var);
  }

protected:
  std::string constraint_name_;
  std::string collision_link_name_;
  CaSX collision_fk_;
};

class FabCapsuleSphereLeaf : public FabGenericGeometryLeaf {
public:
  FabCapsuleSphereLeaf() = default;

  FabCapsuleSphereLeaf(FabVariablesPtr parent_vars, std::string capsule_name, std::string sphere_name,
                       const CaSX& capsule_center_1, const CaSX& capsule_center_2)
      : FabGenericGeometryLeaf(std::move(parent_vars), capsule_name + "_" + sphere_name + "_leaf"),
        capsule_name_(std::move(capsule_name)),
        sphere_name_(std::move(sphere_name)),
        capsule_centers_({capsule_center_1, capsule_center_2}) {
    set_forward_map();
  }

private:
  void set_forward_map() {
    auto capsule_radius_name = std::string("radius_") + capsule_name_;
    const CaSX capsule_radius_var = get_parent_var_param(capsule_radius_name, 1);

    auto sphere_radius_name = std::string("radius_") + sphere_name_;
    const CaSX sphere_radius_var = get_parent_var_param(sphere_radius_name, 1);

    auto sphere_center_name = std::string("x_") + sphere_name_;
    const auto obstacle_dim = capsule_centers_[0].size().first;
    const CaSX sphere_center_var = get_parent_var_param(sphere_center_name, obstacle_dim);

    parent_vars_->add_parameters({
        {std::move(sphere_radius_name), sphere_radius_var},
        {std::move(capsule_radius_name), capsule_radius_var},
        {std::move(sphere_center_name), sphere_center_var},
    });

    diffmap_ = std::make_shared<FabCapsuleSphereMap>(parent_vars_, capsule_centers_, sphere_center_var,
                                                     capsule_radius_var, sphere_radius_var);
  }

protected:
  std::string capsule_name_;
  std::string sphere_name_;
  std::vector<CaSX> capsule_centers_;
};

class FabCapsuleCuboidLeaf : public FabGenericGeometryLeaf {
public:
  FabCapsuleCuboidLeaf() = default;

  FabCapsuleCuboidLeaf(FabVariablesPtr parent_vars, std::string capsule_name, std::string cuboid_name,
                       const CaSX& capsule_center_1, const CaSX& capsule_center_2)
      : FabGenericGeometryLeaf(std::move(parent_vars), capsule_name + "_" + cuboid_name + "_leaf"),
        capsule_name_(std::move(capsule_name)),
        cuboid_name_(std::move(cuboid_name)),
        capsule_centers_({capsule_center_1, capsule_center_2}) {
    set_forward_map();
  }

private:
  void set_forward_map() {
    auto capsule_radius_name = std::string("radius_") + capsule_name_;
    const CaSX capsule_radius_var = get_parent_var_param(capsule_radius_name, 1);

    const auto cuboid_dim = capsule_centers_[0].size().first;
    auto cuboid_size_name = std::string("size_") + cuboid_name_;
    const CaSX cuboid_size_var = get_parent_var_param(cuboid_size_name, cuboid_dim);

    auto cuboid_center_name = std::string("x_") + cuboid_name_;
    const CaSX cuboid_center_var = get_parent_var_param(cuboid_center_name, cuboid_dim);

    parent_vars_->add_parameters({
        {std::move(cuboid_size_name), cuboid_size_var},
        {std::move(capsule_radius_name), capsule_radius_var},
        {std::move(cuboid_center_name), cuboid_center_var},
    });

    diffmap_ = std::make_shared<FabCapsuleSphereMap>(parent_vars_, capsule_centers_, cuboid_center_var,
                                                     capsule_radius_var, cuboid_size_var);
  }

protected:
  std::string capsule_name_;
  std::string cuboid_name_;
  std::vector<CaSX> capsule_centers_;
};

/*
 * Leaf for geometry of a cuboid (3D) obstacle with respect to the collision sphere
 */
class FabSphereCuboidLeaf : public FabGenericGeometryLeaf {
public:
  FabSphereCuboidLeaf() = default;

  FabSphereCuboidLeaf(FabVariablesPtr parent_vars, const CaSX& fk, const std::string& obstacle_name,
                      const std::string& collision_link_name)
      : FabGenericGeometryLeaf(std::move(parent_vars), obstacle_name + "_" + collision_link_name + "_leaf",
                               fk) {
    set_forward_map(obstacle_name, collision_link_name);
  }

private:
  void set_forward_map(const std::string& obstacle_name, const std::string& collision_link_name) {
    const auto radius_body_name = std::string("radius_body_") + collision_link_name;
    const CaSX radius_body_var = get_parent_var_param(radius_body_name, 1);

    // const auto obstacle_dim = forward_kinematics_.size().first;
    const auto cuboid_size_name = std::string("size_") + obstacle_name;
    const CaSX cuboid_size_var = get_parent_var_param(cuboid_size_name, 3);

    const auto cuboid_center_name = std::string("x_") + obstacle_name;
    const CaSX cuboid_center_var = get_parent_var_param(cuboid_center_name, 3);

    parent_vars_->add_parameters({{radius_body_name, radius_body_var},
                                  {cuboid_size_name, cuboid_size_var},
                                  {cuboid_center_name, cuboid_center_var}});

    const auto& sphere_center_var = forward_kinematics_;
    diffmap_ = std::make_shared<FabCuboidSphereMap>(parent_vars_, sphere_center_var, cuboid_center_var,
                                                    radius_body_var, cuboid_size_var);
  }
};