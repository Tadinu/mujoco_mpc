#pragma once

#include <mujoco/mujoco.h>

#include <casadi/casadi.hpp>
#include <cassert>
#include <map>
#include <memory>
#include <shared_mutex>
#include <stdexcept>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_config.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_energized_geometry.h"
#include "mjpc/planners/fabrics/include/fab_energy.h"
#include "mjpc/planners/fabrics/include/fab_forward_kinematics.h"
#include "mjpc/planners/fabrics/include/fab_geometry.h"
#include "mjpc/planners/fabrics/include/fab_goal.h"
#include "mjpc/planners/fabrics/include/fab_robot.h"
#include "mjpc/planners/fabrics/include/fab_spectral_semi_sprays.h"
#include "mjpc/planners/fabrics/include/fab_speed_control.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"
#include "mjpc/planners/fabrics/include/leaf/fab_attractor_leaf.h"
#include "mjpc/planners/fabrics/include/leaf/fab_dynamic_attractor_leaf.h"
#include "mjpc/planners/fabrics/include/leaf/fab_dynamic_geometry_leaf.h"
#include "mjpc/planners/fabrics/include/leaf/fab_dynamic_leaf.h"
#include "mjpc/planners/fabrics/include/leaf/fab_geometry_leaf.h"
#include "mjpc/planners/fabrics/include/leaf/fab_leaf.h"
#include "mjpc/planners/planner.h"
#include "mjpc/utilities.h"

enum class FabControlMode : uint8_t { VEL, ACC };

class FabPlanner : public mjpc::Planner {
public:
  FabPlanner() = default;
  ~FabPlanner() override = default;

  FabVariablesPtr vars() const { return vars_; }
  FabPlannerConfig config() { return config_; }

  void init_robot(int dof, std::string urdf_path, std::string base_link_name,
                  std::vector<std::string> endtip_names) {
    // NOTE: Always need to reset robot afresh regardless to renew its vars
    robot_ = std::make_shared<FabRobot>(dof, std::move(urdf_path), std::move(base_link_name),
                                        std::move(endtip_names));
    vars_ = robot_->vars();
    geometry_ = robot_->weighted_geometry();
    forced_geometry_ = nullptr;
    target_velocity_ = CaSX::zeros(dof);
    cafunc_ = nullptr;
  }

  void add_geometry(const FabDifferentialMap& forward_map, FabLagrangianPtr lagrangian,
                    FabGeometryPtr geometry) {
    add_weighted_geometry(forward_map,
                          FabWeightedSpec({{"g", std::move(geometry)}, {"le", std::move(lagrangian)}}));
  }

  void add_dynamic_geometry(const FabDifferentialMap& forward_map,
                            const FabDynamicDifferentialMap& dynamic_map,
                            const FabDifferentialMap& geometry_map, FabLagrangianPtr lagrangian,
                            FabGeometryPtr geometry) {
    const auto weighted_spec = FabWeightedSpec(
        {{"g", std::move(geometry)}, {"le", std::move(lagrangian)}, {"ref_names", dynamic_map.ref_names()}});
    const auto pwg1 = weighted_spec.pull_back<FabWeightedSpec>(geometry_map);
    const auto pwg2 = pwg1->dynamic_pull_back<FabWeightedSpec>(dynamic_map);
    const auto pwg3 = pwg2->pull_back<FabWeightedSpec>(forward_map);
    *geometry_ += *pwg3;
  }

  void add_weighted_geometry(const FabDifferentialMap& forward_map,
                             const FabWeightedSpec& weighted_geometry) {
    const auto pulled_geometry = weighted_geometry.pull_back<FabWeightedSpec>(forward_map);
    *geometry_ += *pulled_geometry;
    *vars_ += *pulled_geometry->vars();
  }

  void add_leaf(const FabLeaf* leaf, bool is_prime_leaf = false) {
    assert(leaf);

    if (const auto* attractor = dynamic_cast<const FabGenericAttractorLeaf*>(leaf)) {
      add_forcing_geometry(attractor->map(), attractor->lagrangian(), attractor->geometry(), is_prime_leaf);
    } else if (const auto* dyn_attractor = dynamic_cast<const FabGenericDynamicAttractorLeaf*>(leaf)) {
      add_dynamic_forcing_geometry(dyn_attractor->map(), *dyn_attractor->dynamic_map(),
                                   dyn_attractor->lagrangian(), dyn_attractor->geometry(),
                                   dyn_attractor->xdot_ref(), is_prime_leaf);
    } else if (const auto* geom_leaf = dynamic_cast<const FabGenericGeometryLeaf*>(leaf)) {
      add_geometry(*geom_leaf->map(), geom_leaf->lagrangian(), geom_leaf->geometry());
    } else if (const auto* dyn_geom_leaf = dynamic_cast<const FabGenericDynamicGeometryLeaf*>(leaf)) {
      add_dynamic_geometry(*dyn_geom_leaf->map(), *dyn_geom_leaf->dynamic_map(),
                           *dyn_geom_leaf->geometry_map(), dyn_geom_leaf->lagrangian(),
                           dyn_geom_leaf->geometry());
    }

    leaves_.insert_or_assign(leaf->name(), *leaf);
  }

  std::vector<FabLeaf> get_leaves(const std::vector<std::string>& leaf_names) const {
    std::vector<FabLeaf> out_leaves;
    for (const auto& leaf_name : leaf_names) {
      if (!leaves_.contains(leaf_name)) {
        throw FabError("Leaf not found: " + leaf_name +
                       "\nPossible leaves: " + fab_core::join(fab_core::get_map_keys(leaves_), ";"));
      }
      out_leaves.push_back(leaves_.at(leaf_name));
    }
    return out_leaves;
  }

  void add_forcing_geometry(FabDifferentialMapPtr forward_map, FabLagrangianPtr lagrangian,
                            const FabGeometryPtr& geometry, bool is_prime_forcing_leaf) {
    if (nullptr == forced_geometry_) {
      forced_geometry_ = std::make_shared<FabWeightedSpec>(*geometry_);
    }
    *forced_geometry_ += *FabWeightedSpec({{"g", geometry}, {"le", std::move(lagrangian)}})
                              .pull_back<FabWeightedSpec>(*forward_map);
    if (is_prime_forcing_leaf) {
      forced_vars_ = geometry->vars();
      forced_forward_map_ = std::move(forward_map);
    }
    *vars_ += *forced_geometry_->vars();
    geometry_->concretize();
    forced_geometry_->concretize(ref_sign_);
  }

  void add_dynamic_forcing_geometry(const FabDifferentialMapPtr& forward_map,
                                    const FabDynamicDifferentialMap& dynamic_map, FabLagrangianPtr lagrangian,
                                    const FabGeometryPtr& geometry, const CaSX& target_velocity,
                                    bool is_prime_forcing_leaf) {
    if (nullptr == forced_geometry_) {
      forced_geometry_ = std::make_shared<FabWeightedSpec>(*geometry_);
    }
    const auto wg = FabWeightedSpec({{"g", geometry}, {"le", std::move(lagrangian)}})
                        .pull_back<FabWeightedSpec>(*forward_map);
    const auto pwg = wg->dynamic_pull_back<FabWeightedSpec>(dynamic_map);
    const auto ppwg = pwg->pull_back<FabWeightedSpec>(*forward_map);
    *forced_geometry_ += *ppwg;
    if (is_prime_forcing_leaf) {
      forced_vars_ = geometry->vars();
      forced_forward_map_ = forward_map;
    }
    *vars_ += *forced_geometry_->vars();
    target_velocity_ += CaSX::mtimes(forward_map->J().T(), target_velocity);
    ref_sign_ = -1;
    geometry_->concretize();
    forced_geometry_->concretize(ref_sign_);
  }

  void set_execution_energy(FabLagrangianPtr execution_lagrangian) {
    execution_lagrangian_ = std::move(execution_lagrangian);
    execution_geometry_ = std::make_shared<FabWeightedSpec>(
        FabWeightedSpecArgs{{"g", std::make_shared<FabGeometry>(FabGeometryArgs{{"s", geometry_}})},
                            {"le", execution_lagrangian_}});
    execution_geometry_->concretize();

    try {
      forced_speed_controlled_geometry_ = std::make_shared<FabWeightedSpec>(
          FabWeightedSpecArgs{{"g", std::make_shared<FabGeometry>(FabGeometryArgs{{"s", forced_geometry_}})},
                              {"le", execution_lagrangian_}});
      forced_speed_controlled_geometry_->concretize();
    } catch (const FabParamNotFoundError& e) {
      FAB_PRINT(e.what());
      assert(false);
    }
  }

  void set_speed_control() {
    const auto& x_psi = forced_vars_->position_var();
    const auto& dm_psi = forced_forward_map_;
    assert(execution_lagrangian_);
    const auto& ex_lag = execution_lagrangian_;
    const auto a_ex = CaSX::sym("a_ex_damper", 1);
    const auto a_le = CaSX::sym("a_le_damper", 1);
    damper_ = FabDamper(x_psi, config_.damper_beta(x_psi, a_ex, a_le), a_ex, a_le,
                        config_.damper_eta(CaSX::vertcat(CaSX::symvar(ex_lag->l()))), dm_psi, ex_lag->l());
  }

  CaSX get_forward_kinematics(const std::string& link_name, bool position_only = true) {
    const auto fk = robot_->fk();
    assert(fk);
    return fk ? fk->casadi(vars_->position_var(), link_name, {}, fab_math::CASX_TRANSF_IDENTITY,
                           position_only)
              : CaSX();
  }

  void add_capsule_sphere_geometry(const std::string& obstacle_name, const std::string& capsule_name,
                                   const CaSX& tf_capsule_origin, double capsule_length) {
    const auto capsule_radius = 0.5 * capsule_length;
    auto tf_origin_center_0 = fab_math::CASX_TRANSF_IDENTITY;
    fab_core::set_casx2(tf_origin_center_0, 2, 3, capsule_radius);
    auto tf_origin_center_1 = fab_math::CASX_TRANSF_IDENTITY;
    fab_core::set_casx2(tf_origin_center_1, 2, 3, -capsule_radius);
    const auto tf_center_0 = CaSX::mtimes(tf_capsule_origin, tf_origin_center_0);
    const auto tf_center_1 = CaSX::mtimes(tf_capsule_origin, tf_origin_center_1);
    auto capsule_sphere_leaf =
        FabCapsuleSphereLeaf(vars_, capsule_name, obstacle_name, fab_core::get_casx2(tf_center_0, {0, 3}, 3),
                             fab_core::get_casx2(tf_center_1, {0, 3}, 3));
    capsule_sphere_leaf.set_geometry(config_.collision_geometry);
    capsule_sphere_leaf.set_finsler_structure(config_.collision_finsler);
    add_leaf(&capsule_sphere_leaf);
  }

  void add_capsule_cuboid_geometry(const std::string& obstacle_name, const std::string& capsule_name,
                                   const CaSX& tf_capsule_origin, double capsule_length) {
    const auto capsule_radius = 0.5 * capsule_length;
    auto tf_origin_center_0 = fab_math::CASX_TRANSF_IDENTITY;
    fab_core::set_casx2(tf_origin_center_0, 2, 3, capsule_radius);
    auto tf_origin_center_1 = fab_math::CASX_TRANSF_IDENTITY;
    fab_core::set_casx2(tf_origin_center_1, 2, 3, -capsule_radius);
    const auto tf_center_0 = CaSX::mtimes(tf_capsule_origin, tf_origin_center_0);
    const auto tf_center_1 = CaSX::mtimes(tf_capsule_origin, tf_origin_center_1);
    auto capsule_cuboid_leaf =
        FabCapsuleCuboidLeaf(vars_, capsule_name, obstacle_name, fab_core::get_casx2(tf_center_0, {0, 3}, 3),
                             fab_core::get_casx2(tf_center_1, {0, 3}, 3));
    capsule_cuboid_leaf.set_geometry(config_.collision_geometry);
    capsule_cuboid_leaf.set_finsler_structure(config_.collision_finsler);
    add_leaf(&capsule_cuboid_leaf);
  }

  /*
   * [fk] should be a symbolic expression using CasADi SX for the obstacle's position.
   */
  void add_spherical_obstacle_geometry(const std::string& obstacle_name,
                                       const std::string& collision_link_name, const CaSX& fk) {
    auto spherical_obstacle_leaf = FabObstacleLeaf(vars_, fk, obstacle_name, collision_link_name);
    spherical_obstacle_leaf.set_geometry(config_.collision_geometry);
    spherical_obstacle_leaf.set_finsler_structure(config_.collision_finsler);
    add_leaf(&spherical_obstacle_leaf);
  }

  void add_dynamic_spherical_obstacle_geometry(const std::string& obstacle_name,
                                               const std::string& collision_link_name, const CaSX& fk,
                                               const CaSXDict& reference_params,
                                               int dynamic_obstacle_dimension = 3) {
    assert(dynamic_obstacle_dimension <= fk.size().first);
    auto dyn_spherical_obstacle_leaf = FabDynamicObstacleLeaf(
        vars_, fab_core::get_casx(fk, std::array<casadi_int, 2>{0, casadi_int(dynamic_obstacle_dimension)}),
        obstacle_name, collision_link_name, reference_params);
    dyn_spherical_obstacle_leaf.set_geometry(config_.collision_geometry);
    dyn_spherical_obstacle_leaf.set_finsler_structure(config_.collision_finsler);
    add_leaf(&dyn_spherical_obstacle_leaf);
  }

  void add_plane_constraint(std::string constraint_name, std::string collision_link_name, const CaSX& fk) {
    auto plane_constraint =
        FabPlaneConstraintGeometryLeaf(vars_, std::move(constraint_name), std::move(collision_link_name), fk);
    plane_constraint.set_geometry(config_.geometry_plane_constraint);
    plane_constraint.set_finsler_structure(config_.finsler_plane_constraint);
    add_leaf(&plane_constraint);
  }

  void add_cuboid_obstacle_geometry(const std::string& obstacle_name, const std::string& collision_link_name,
                                    const CaSX& fk) {
    auto cuboid_obstacle_leaf = FabSphereCuboidLeaf(vars_, fk, obstacle_name, collision_link_name);
    cuboid_obstacle_leaf.set_geometry(config_.collision_geometry);
    cuboid_obstacle_leaf.set_finsler_structure(config_.collision_finsler);
    add_leaf(&cuboid_obstacle_leaf);
  }

  void add_esdf_geometry(std::string collision_link_name) {
    const CaSX fk = get_forward_kinematics(collision_link_name);
    auto geometry = FabESDFGeometryLeaf(vars_, std::move(collision_link_name), fk);
    geometry.set_geometry(config_.collision_geometry);
    geometry.set_finsler_structure(config_.collision_finsler);
    add_leaf(&geometry);
  }

  void add_spherical_self_collision_geometry(const std::string& collision_link_1_name,
                                             const std::string& collision_link_2_name) {
    const auto fk_1 = get_forward_kinematics(collision_link_1_name);
    const auto fk_2 = get_forward_kinematics(collision_link_2_name);
    const auto fk = fk_2 - fk_1;
    if (fab_core::is_casx_sparse(fk)) {
      FAB_PRINT(std::string("Expression [") + fk.get_str() + "] for links " + collision_link_1_name + "and " +
                collision_link_2_name + " is sparse and so skipped");
      auto geometry = FabSelfCollisionLeaf(vars_, fk, collision_link_1_name, collision_link_2_name);
      geometry.set_geometry(config_.self_collision_geometry);
      geometry.set_finsler_structure(config_.self_collision_finsler);
      add_leaf(&geometry);
    }
  }

  void add_limit_geometry(const int joint_index, const FabJointLimit& limits) {
    auto lower_limit_geometry = FabLimitLeaf(vars_, joint_index, limits[0], 0);
    lower_limit_geometry.set_geometry(config_.limit_geometry);
    lower_limit_geometry.set_finsler_structure(config_.limit_finsler);
    auto upper_limit_geometry = FabLimitLeaf(vars_, joint_index, limits[1], 1);
    upper_limit_geometry.set_geometry(config_.limit_geometry);
    upper_limit_geometry.set_finsler_structure(config_.limit_finsler);
    add_leaf(&lower_limit_geometry);
    add_leaf(&upper_limit_geometry);
  }

  void load_problem_configuration(FabPlannerProblemConfig problem_config) {
    problem_config_ = std::move(problem_config);
    for (const auto& obstacle : problem_config_.environment().obstacles()) {
      vars_->add_parameters(obstacle->sym_parameters());
    }

    set_collision_avoidance();
    set_self_collision_avoidance();
    set_joint_limits();

    // [Goal composition]
    if (fab_core::has_collection_element(
            std::array<FORCING_TYPE, 3>{FORCING_TYPE::SPEED_CONTROLLED, FORCING_TYPE::FORCED,
                                        FORCING_TYPE::FORCED_ENERGIZED},
            config_.forcing_type)) {
      set_goal_component(problem_config_.goal_composition());
    }

    // [Execution Energy]
    if (fab_core::has_collection_element(
            std::array<FORCING_TYPE, 3>{FORCING_TYPE::SPEED_CONTROLLED, FORCING_TYPE::EXECUTION_ENERGY,
                                        FORCING_TYPE::FORCED_ENERGIZED},
            config_.forcing_type)) {
      set_execution_energy(std::make_shared<FabExecutionLagrangian>(vars_));
    }

    // [SPEED CONTROL]
    if (config_.forcing_type == FORCING_TYPE::SPEED_CONTROLLED) {
      set_speed_control();
    }
  }

  void set_joint_limits() {
    const auto joint_limits = problem_config_.joint_limits();
    assert(joint_limits.lower_limits.size() == joint_limits.upper_limits.size());
    for (auto i = 0; i < joint_limits.lower_limits.size(); ++i) {
      add_limit_geometry(i, {joint_limits.lower_limits[i], joint_limits.upper_limits[i]});
    }
  }

  void set_self_collision_avoidance() {
    const auto robot_rep = problem_config_.robot_representation();
    const auto self_collision_pairs = robot_rep.self_collision_pairs();
    for (const auto& [link_name, pair_links_names] : self_collision_pairs) {
      const auto link = robot_rep.collision_link(link_name);
      if (dynamic_pointer_cast<FabSphere>(link)) {
        for (const auto& paired_link_name : pair_links_names) {
          const auto paired_link = robot_rep.collision_link(paired_link_name);
          if (dynamic_pointer_cast<FabSphere>(paired_link)) {
            add_spherical_self_collision_geometry(paired_link_name, link_name);
          }
        }
      }
    }
  }

  void set_collision_avoidance() {
    const auto environment = problem_config_.environment();
    const auto robot_representation = problem_config_.robot_representation();
    const auto collision_links = robot_representation.collision_links();
    for (const auto& [link_name, collision_link] : collision_links) {
      CaSX fk = get_forward_kinematics(link_name, false);
      const auto fk_size = fk.size();
      if (fk_size == decltype(fk_size){3, 3}) {
        auto fk_augmented = fab_math::CASX_TRANSF_IDENTITY;
        fab_core::set_casx2(fk_augmented, {0, 2}, {0, 2}, fab_core::get_casx2(fk, {0, 2}, {0, 2}));
        fab_core::set_casx2(fk_augmented, {0, 2}, 3, fab_core::get_casx2(fk, {0, 2}, 2));
        fk = fk_augmented;
      }
      if (fk_size == decltype(fk_size){4, 4}) {
        const auto fk_0_3_3 = fab_core::get_casx2(fk, {0, 3}, 3);
        if (fab_core::is_casx_sparse(fk_0_3_3)) {
          FAB_PRINT(std::string("Expression ") + fk_0_3_3.get_str() + " for link " + link_name +
                    " is sparse and so skipped");
          continue;
        }
      } else if (fab_core::is_casx_sparse(fk)) {
        FAB_PRINT(std::string("Expression ") + fk.get_str() + " for link " + link_name +
                  " is sparse and so skipped");
        continue;
      }

      collision_link->set_origin(fk);
      vars_->add_parameters(collision_link->sym_parameters());
      vars_->add_parameter_values(collision_link->parameters());
      for (const auto& obstacle : environment.obstacles()) {
        const CaSX distance = collision_link->distance(obstacle.get());
        const auto leaf_name = link_name + "_" + obstacle->name() + "_leaf";
        auto leaf = FabAvoidanceLeaf(vars_, leaf_name, distance);
        leaf.set_geometry(config_.collision_geometry);
        leaf.set_finsler_structure(config_.collision_finsler);
        add_leaf(&leaf);
      }
#if 0
      for (auto i = 0; i < environment.spheres_num(); ++i) {
        const auto obstacle_name = std::string("obst_sphere_") + std::to_string(i);
        if (dynamic_pointer_cast<FabSphere>(collision_link)) {
          add_dynamic_spherical_obstacle_geometry(obstacle_name, link_name, fk, reference_parameter_list[i],
                                                  dynamic_obstacle_dimension);
        }
      }
      for (auto i = 0; i < environment.planes_num(); ++i) {
        const auto constraint_name = std::string("constraint_") + std::to_string(i);
        if (dynamic_pointer_cast<FabSphere>(collision_link)) {
          add_plane_constraint(constraint_name, link_name, fk);
        }
      }

      for (auto i = 0; i < environment.cuboids_num(); ++i) {
        const auto obstacle_name = std::string("obst_cuboid_") + std::to_string(i);
        if (dynamic_pointer_cast<FabSphere>(collision_link)) {
          add_cuboid_obstacle_geometry(obstacle_name, link_name, fk);
        }
      }
#endif
    }
  }

  void set_components(const std::vector<std::string>& collision_link_names,
                      const FabSelfCollisionPairs& self_collision_pairs,
                      const std::vector<std::string>& collision_links_esdf, const FabGoalComposition& goal,
                      const std::vector<FabJointLimit>& limits, int static_obstacles_num = 1,
                      int dynamic_obstacles_num = 0, int cuboid_obstacles_num = 0,
                      int plane_constraints_num = 1, int dynamic_obstacle_dimension = 3) {
    // Obstacles
    std::vector<CaSXDict> reference_param_list;
    for (auto i = 0; i < dynamic_obstacles_num; ++i) {
      CaSXDict reference_params = {
          {std::string("x_obst_dynamic_") + std::to_string(i),
           CaSX::sym(std::string("x_obst_dynamic_") + std::to_string(i), dynamic_obstacle_dimension)},
          {std::string("xdot_obst_dynamic_") + std::to_string(i),
           CaSX::sym(std::string("xdot_obst_dynamic_") + std::to_string(i), dynamic_obstacle_dimension)},
          {std::string("xddot_obst_dynamic_") + std::to_string(i),
           CaSX::sym(std::string("xddot_obst_dynamic_") + std::to_string(i), dynamic_obstacle_dimension)}};
      reference_param_list.emplace_back(std::move(reference_params));
    }

    // Collision link geoms
    for (const auto& link_name : collision_link_names) {
      const auto fk = get_forward_kinematics(link_name);
      if (fab_core::is_casx_sparse(fk)) {
        FAB_PRINT(std::string("Expression ") + fk.get_str() + " for link " + link_name +
                  "is sparse and so skipped");
        continue;
      }

      for (auto i = 0; i < static_obstacles_num; ++i) {
        const auto obstacle_name = std::string("obst_") + std::to_string(i);
        add_spherical_obstacle_geometry(obstacle_name, link_name, fk);
      }
      for (auto i = 0; i < dynamic_obstacles_num; ++i) {
        const auto obstacle_name = std::string("obst_dynamic_") + std::to_string(i);
        add_dynamic_spherical_obstacle_geometry(obstacle_name, link_name, fk, reference_param_list[i],
                                                dynamic_obstacle_dimension);
      }
      for (auto i = 0; i < plane_constraints_num; ++i) {
        const auto constraint_name = std::string("constraint_") + std::to_string(i);
        add_plane_constraint(constraint_name, link_name, fk);
      }

      for (auto i = 0; i < cuboid_obstacles_num; ++i) {
        const auto obstacle_name = std::string("obst_cuboid_") + std::to_string(i);
        add_cuboid_obstacle_geometry(obstacle_name, link_name, fk);
      }
    }

    // Collision-links ESDF
    for (const auto& link_name : collision_links_esdf) {
      add_esdf_geometry(link_name);
    }

    // Self-collision link-pairs
    for (const auto& [link_name_1, link_pair] : self_collision_pairs) {
      for (const auto& link_name_2 : link_pair) {
        add_spherical_self_collision_geometry(link_name_2, link_name_1);
      }
    }

    // Joint limits
    if (!limits.empty()) {
      for (auto i = 0; i < limits.size(); ++i) {
        add_limit_geometry(i, limits[i]);
      }
    }

    // Execution energy
    if (goal.is_valid()) {
      set_goal_component(goal);
      set_execution_energy(std::make_shared<FabExecutionLagrangian>(vars_));
      set_speed_control();
    } else {
      set_execution_energy(std::make_shared<FabExecutionLagrangian>(vars_));
    }
  }

  CaSX get_differential_map(const FabSubGoalPtr& sub_goal) {
    const auto subgoal_type = sub_goal->type();
    const auto subgoal_indices = sub_goal->indices();
    if (FabSubGoalType::STATIC_JOINT_SPACE == subgoal_type) {
      return fab_core::get_casx(vars_->position_var(), subgoal_indices);
    } else {
      const auto fk_child = get_forward_kinematics(sub_goal->child_link_name());
      CaSX fk_parent;
      try {
        fk_parent = get_forward_kinematics(sub_goal->parent_link_name());
      } catch (const FabError& e) {
        fk_parent = CaSX::zeros(3);
      }
      FAB_PRINTDB("fk_child", sub_goal->child_link_name(), fab_core::get_casx(fk_child, subgoal_indices),
                  subgoal_indices);
      FAB_PRINTDB("fk_parent", sub_goal->parent_link_name(), fab_core::get_casx(fk_parent, subgoal_indices),
                  subgoal_indices);
      return fab_core::get_casx(fk_child, subgoal_indices) - fab_core::get_casx(fk_parent, subgoal_indices);
    }
  }

  void set_goal_component(const FabGoalComposition& goal) {
    const auto sub_goals = goal.sub_goals();
    for (auto i = 0; i < sub_goals.size(); ++i) {
      const auto& sub_goal = sub_goals[i];
      const auto fk_sub_goal = get_differential_map(sub_goal);
      if (fab_core::is_casx_sparse(fk_sub_goal)) {
        throw FabError(fk_sub_goal.get_str() + "must not be sparse");
      }

      // [Goal attractor leaf]
      const auto goal_name = std::string("goal_") + std::to_string(i);
      std::unique_ptr<FabLeaf> attractor = nullptr;
      if (FabSubGoalType::DYNAMIC == sub_goal->type()) {
        attractor = std::make_unique<FabGenericDynamicAttractorLeaf>(vars_, fk_sub_goal, goal_name);
      } else {
        attractor = std::make_unique<FabGenericAttractorLeaf>(vars_, fk_sub_goal, goal_name);
      }
      attractor->set_potential(config_.attractor_potential, sub_goal->weight());
      attractor->set_metric(config_.attractor_metric);
      add_leaf(attractor.get(), sub_goal->is_primary_goal());
    }
  }

  void concretize(const FabControlMode control_mode = FabControlMode::ACC, float time_step = 0) {
    control_mode_ = control_mode;
    if ((FabControlMode::VEL == control_mode) && (time_step < 0)) {
      throw FabError(std::to_string(time_step) + ": Invalid time step passed in velocity control mode");
    }
    geometry_->concretize();

    // xddot
    CaSX xddot;
    switch (config_.forcing_type) {
      case FORCING_TYPE::SPEED_CONTROLLED: {
        const CaSX eta = damper_.substitute_eta();
        const CaSX a_ex =
            (eta * execution_geometry_->alpha()) + ((1 - eta) * forced_speed_controlled_geometry_->alpha());
        const CaSX beta_subst = damper_.substitute_beta(-a_ex, -geometry_->alpha());
#if 1
        xddot = forced_geometry_->xddot() -
                (a_ex + beta_subst) *
                    (geometry_->xdot() - CaSX::mtimes(forced_geometry_->Minv(), target_velocity_));
#else
        xddot = forced_geometry_.xddot();
#endif
      } break;

      case FORCING_TYPE::EXECUTION_ENERGY: {
        FAB_PRINT("No forcing term, using pure geometry with energization");
#if 1
        xddot =
            execution_geometry_->xddot() - execution_geometry_->alpha() * geometry_->vars()->velocity_var();
#else
        xddot = geometry_->xddot() - geometry_->alpha() * geometry_->vars()->velocity_var();
#endif
      } break;

      case FORCING_TYPE::FORCED_ENERGIZED: {
        FAB_PRINT("Using forced geometry with constant execution energy");
        xddot = forced_speed_controlled_geometry_->xddot() -
                forced_speed_controlled_geometry_->alpha() * geometry_->vars()->velocity_var();
      } break;

      case FORCING_TYPE::FORCED: {
        FAB_PRINT("No execution energy, using forced geometry without speed regulation");
        xddot = forced_geometry_->xddot() - geometry_->alpha() * geometry_->vars()->velocity_var();
      } break;

      case FORCING_TYPE::PURE_GEOMETRY: {
        xddot = geometry_->xddot();
      } break;

      default:
        throw FabError(std::to_string(int(config_.forcing_type)) + " :Unknown forcing type");
    }  // end switch(config_.forcing_type)

    // CasadiFunction
    switch (control_mode) {
      case FabControlMode::ACC: {
        cafunc_ =
            std::make_shared<FabCasadiFunction>("fab_planner_func", *vars_, CaSXDict{{"action", xddot}});
      } break;

      case FabControlMode::VEL: {
        assert(time_step > 0);
        cafunc_ = std::make_shared<FabCasadiFunction>(
            "fab_planner_func", *vars_, CaSXDict{{"action", geometry_->xdot() + time_step * xddot}});
      } break;
    }
  }

  /*
   * Computes action based on the states passed.
   * The variables passed are the joint states, and the goal position.
   * The action is nullified if its magnitude is very large or very small.
   */
  CaSX compute_action(const FabCasadiArgMap& kwargs) const {
    if (!cafunc_) {
      return {};
    }
    // const FabSharedMutexLock lock(policy_mutex_);
    auto eval = cafunc_->evaluate(kwargs);
    CaSX action = eval["action"];
    if (!action.is_zero()) {
      // Debugging
      FAB_PRINTDB("a_ex: ", eval["a_ex"]);
      FAB_PRINTDB("alpha_forced_geometry:", eval["alpha_forced_geometry"]);
      FAB_PRINTDB("alpha_geometry:", eval["alpha_geometry"]);
      FAB_PRINTDB("beta", eval["beta"]);

      const double action_magnitude = double(CaSX::norm_2(action).scalar());
      if (action_magnitude < casadi::eps) {
        FAB_PRINT("Fabrics: Avoiding SMALL action with magnitude", action_magnitude);
        action = 0.0;
      } else if (action_magnitude > (1 / casadi::eps)) {
        FAB_PRINT("Fabrics: Avoiding LARGE action with magnitude", action_magnitude);
        action = 0.0;
      }
    }
    return action;
  }

  // =========================================================================================================
  // MJPC-PLANNER IMPL --
  //
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  mjpc::Task* task_ = nullptr;

  urdf::UrdfModel RobotURDFModel() const override {
    const auto urdf_fk = robot_ ? robot_->fk() : nullptr;
    return urdf_fk ? dynamic_pointer_cast<FabURDFForwardKinematics>(urdf_fk)->urdf_model()
                   : urdf::UrdfModel();
  }

  // initialize data and settings
  void Initialize(mjModel* model, const mjpc::Task& task) override {
    task_ = const_cast<mjpc::Task*>(&task);
    model_ = model;
    data_ = task_->data_;

    // dimensions
    dim_state_ = model->nq + model->nv + model->na;     // state dimension
    dim_state_derivative_ = 2 * model->nv + model->na;  // state derivative dimension
    dim_action_ = task.GetActionDim();                  // action dimension
    dim_sensor_ = model->nsensordata;                   // number of sensor values
    dim_max_ = std::max({dim_state_, dim_state_derivative_, dim_action_, model->nuser_sensor});

    if (trajectory_) {
      trajectory_->Reset(0);
    } else {
      trajectory_ = std::make_shared<mjpc::Trajectory>();
    }

    // Init task fabrics
    if (task.IsFabricsSupported()) {
      // Robot, resetting [vars_, geometry_, target_velocity_] here-in!
      init_robot(dim_action_, task.URDFPath(), task.GetBaseBodyName(), task.GetEndtipNames());

      // Config
      config_ = task.GetFabricsConfig(task.IsGoalFixed() && task.AreObstaclesFixed());

      // Goal
      FabGoalComposition goal;
      for (const auto& subgoal : task.GetSubGoals()) {
        goal.add_sub_goal(subgoal);
      }

      // Add geometry + energy components + [goal]
      set_components(task_->GetCollisionLinkNames(), {}, {}, goal,
                     task_->GetJointLimits() /* TODO: Fetch from RobotURDFModel()->joint_map*/,
                     task_->AreObstaclesFixed() ? task_->GetStaticObstaclesNum() : 0,
                     task_->AreObstaclesFixed() ? 0 : task_->GetDynamicObstaclesNum(),
                     0 /*cuboid_obstacles_num*/, task_->GetPlaneConstraintsNum(),
                     task_->GetDynamicObstaclesDim());

      // Concretize, calculating [xddot] + composing [cafunc_] based on it
      concretize(FAB_USE_ACTUATOR_VELOCITY ? FabControlMode::VEL : FabControlMode::ACC, 0.01);
    }
  }

  void Allocate() override {
    trajectory_->Initialize(dim_state_, dim_action_, task_->num_residual, task_->num_trace, 1);
    trajectory_->Allocate(1);
  }

  // reset memory to zeros
  void Reset(int horizon, const double* initial_repeated_action = nullptr) override {}

  void SetState(const mjpc::State& state) override {}

  const mjpc::Trajectory* BestTrajectory() override { return trajectory_.get(); }

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override {
#if FAB_DRAW_TRAJECTORY
    std::vector<double> traces;
    {
      const FabSharedMutexLock lock(policy_mutex_);
      if (trajectory_->trace.size() >= 6) {
        traces = trajectory_->trace;
      }
    }

    static constexpr float GREEN[] = {0.0, 1.0, 0.0, 1.0};
    for (auto i = 0; (!traces.empty()) && (i < (traces.size() / 3) - 1); ++i) {
      mjpc::AddConnector(scn ? scn : task_->scene_, mjGEOM_LINE, 5,
                         (mjtNum[]){traces[3 * i], traces[3 * i + 1], traces[3 * i + 2]},
                         (mjtNum[]){traces[3 * (i + 1)], traces[3 * (i + 1) + 1], traces[3 * (i + 1) + 2]},
                         GREEN);
    }
#endif
  }

  void ClearTrace() override {
    const FabSharedMutexLock lock(policy_mutex_);
    trajectory_->trace.clear();
  }

  // planner-specific GUI elements
  void GUI(mjUI& ui) override {}

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift, int timer_shift, int planning,
             int* shift) override {}

  // return number of parameters optimized by planner
  int NumParameters() override { return 0; }

  // optimize nominal policy
  void OptimizePolicy(int horizon, mjpc::ThreadPool& pool) override {
    // get nominal trajectory
    this->NominalTrajectory(horizon, pool);

    // plan
    plan();
  }

  void plan() {
    const auto robot_dof = robot_->dof();
    std::vector<double> q = task_->QueryJointPos(robot_dof);
    std::vector<double> qdot = task_->QueryJointVel(robot_dof);
    FAB_PRINTDB("QPOS", q);
    FAB_PRINTDB("QVEL", qdot);
    FabCasadiArgMap args = {{"q", std::move(q)}, {"qdot", std::move(qdot)}};

    // [Plane constraints]
    for (auto i = 0; i < task_->GetPlaneConstraintsNum(); ++i) {
      args.insert_or_assign(std::string("constraint_") + std::to_string(i),
                            std::vector<double>{0, 0, 1, 0.0});
    }

    // [X-space goals] & [Weights of goals]
    const auto sub_goals = task_->GetSubGoals();
    for (auto i = 0; i < sub_goals.size(); ++i) {
      const auto& sub_goal = sub_goals[i];
      const auto i_str = std::to_string(i);
      args.insert_or_assign(std::string("x_goal_") + i_str, sub_goal->cfg_.desired_position);
      args.insert_or_assign(std::string("weight_goal_") + i_str, sub_goal->cfg_.weight);
    }

    // [Radius collision bodies]
    for (const auto& collision_link_prop : task_->GetCollisionLinkProps()) {
      args.insert_or_assign(std::string("radius_body_") + collision_link_prop.first,
                            std::vector{collision_link_prop.second});
    }

    // [Obstacles' size & pos]
    const auto obstacle_statesX = task_->GetObstacleStatesX();
    const auto obstacles_num = obstacle_statesX.size();
    const bool fixed_obstacles = task_->AreObstaclesFixed();
    const auto fobstacle_prop_name = [&fixed_obstacles](const char* prefix, const int i) {
      return (fixed_obstacles ? prefix : (std::string(prefix) + "dynamic_")) + std::to_string(i);
    };
    for (auto i = 0; i < obstacles_num; ++i) {
      const auto& obstacle_i = obstacle_statesX[i];
      args.insert_or_assign(fobstacle_prop_name("radius_obst_", i),
                            FAB_OBSTACLE_SIZE_SCALE * obstacle_i.size_[0]);
      args.insert_or_assign(fobstacle_prop_name("x_obst_", i),
                            fixed_obstacles
                                ? std::vector{obstacle_i.pos_[0], obstacle_i.pos_[1], obstacle_i.pos_[2]}
                                : std::vector{obstacle_i.pos_[0], obstacle_i.pos_[1]});
      if (!fixed_obstacles) {
        args.insert_or_assign(fobstacle_prop_name("xdot_obst_", i),
                              std::vector{obstacle_i.vel_[0], obstacle_i.vel_[1]});
        args.insert_or_assign(fobstacle_prop_name("xddot_obst_", i),
                              std::vector{obstacle_i.acc_[0], obstacle_i.acc_[1]});
      }
    }

    // Compute action
    CaSX action = compute_action(args);
    if (!action.is_empty()) {
      const FabSharedMutexLock lock(policy_mutex_);
      action_ = action;
    }
  }

  // compute trajectory using nominal policy

  void NominalTrajectory(int horizon, mjpc::ThreadPool& pool) override {}

  // set action from policy
  void ActionFromPolicy(double* action, const double* state, double time, bool use_previous) override {
    const FabSharedMutexLock lock(policy_mutex_);
    if (robot_) {
      const auto dof = robot_->dof();
      if (dof != action_.size1()) {
        return;
      }

      const mjtNum* cur_pos = task_->GetStartPos();
      if (cur_pos) {
        trajectory_->trace.push_back(cur_pos[0]);
        trajectory_->trace.push_back(cur_pos[1]);
        trajectory_->trace.push_back(cur_pos[2]);
      }
      FAB_PRINT(action_);
      for (auto i = 0; i < dof; ++i) {
#if FAB_USE_ACTUATOR_VELOCITY
        const auto action_i = action_(i);
        action[i] = action_i.is_regular() ? task_->actuator_kv * double(action_i.scalar()) : 0;
        if (action[i] > 0) {
          FAB_PRINT("Action", i, action[i]);
        }
#elif FAB_USE_ACTUATOR_MOTOR
        static const auto pointmass_id = task_->GetTargetObjectId();
        static const auto pointmass = model_->body_mass[pointmass_id];
        static const auto& linear_inertia = pointmass;
        action[i] = action_i.is_regular() ? linear_inertia * double(action_i.scalar()) : 0;
#endif
      }
      // Clamp controls
      mjpc::Clamp(action, model_->actuator_ctrlrange, model_->nu);
    }
  }

protected:
  // NOTE: Each planner can only plan motion of a single robot atm
  FabRobotPtr robot_ = nullptr;
  FabControlMode control_mode_;
  CaSX target_velocity_;
  FabPlannerConfig config_;
  FabPlannerProblemConfig problem_config_;
  FabVariablesPtr vars_ = nullptr;
  FabWeightedSpecPtr geometry_ = nullptr;
  FabWeightedSpecPtr forced_geometry_ = nullptr;
  FabVariablesPtr forced_vars_ = nullptr;
  FabDifferentialMapPtr forced_forward_map_ = nullptr;
  std::map<std::string, FabLeaf> leaves_;
  FabLagrangianPtr execution_lagrangian_ = nullptr;
  FabWeightedSpecPtr execution_geometry_ = nullptr;
  FabWeightedSpecPtr forced_speed_controlled_geometry_ = nullptr;
  FabDamper damper_;
  FabCasadiFunctionPtr cafunc_ = nullptr;
  int8_t ref_sign_ = 1;

  // mjpc
  std::shared_ptr<mjpc::Trajectory> trajectory_ = nullptr;
  int dim_state_ = 0;             // state
  int dim_state_derivative_ = 0;  // state derivative
  int dim_action_ = 0;            // action
  int dim_sensor_ = 0;            // output (i.e., all sensors)
  int dim_max_ = 0;               // maximum dimension
  mutable std::shared_mutex policy_mutex_;
  CaSX action_;
};
