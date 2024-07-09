#pragma once

#include <casadi/casadi.hpp>
#include <cassert>
#include <map>
#include <memory>
#include <stdexcept>
#include <typeinfo>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_energized_geometry.h"
#include "mjpc/planners/fabrics/include/fab_energy.h"
#include "mjpc/planners/fabrics/include/fab_forward_kinematics.h"
#include "mjpc/planners/fabrics/include/fab_geometry.h"
#include "mjpc/planners/fabrics/include/fab_planner_config.h"
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

class FabPlanner : public mjpc::Planner {
 public:
  FabPlanner() = default;

  explicit FabPlanner(int dof, std::shared_ptr<FabForwardKinematics> fk) : dof_(dof), fk_(std::move(fk)) {
    initialize_joint_variables();
    setup_base_geometry();
  }

  FabVariables vars() { return vars_; }
  FabPlannerConfig config() { return config_; }

  void initialize_joint_variables() {
    const auto q = CaSX::sym("q", dof_);
    const auto qdot = CaSX::sym("qdot", dof_);
    vars_ = FabVariables({{"q", q}, {"qdot", qdot}});
  }

  void setup_base_geometry() {
    const auto q = vars_.position_var();
    const auto qdot = vars_.velocity_var();
    // const auto new_parameters, base_energy =  parse_symbolic_input(self._config.base_energy, q, qdot);
    // vars_.add_parameters(new_parameters);
    auto base_geometry = FabGeometry({{"h", CaSX::zeros(dof_)}, {"var", vars_}});
    auto base_lagrangian = FabLagrangian(config_.base_energy(qdot), {{"var", vars_}});
    geometry_ = FabWeightedGeometry({{"g", std::move(base_geometry)}, {"le", std::move(base_lagrangian)}});
    target_velocity_(CaSX::zeros(geometry_.x().size().first));
  }

  void add_geometry(const FabDifferentialMap& forward_map, FabLagrangian lagrangian, FabGeometry geometry) {
    add_weighted_geometry(forward_map,
                          FabWeightedGeometry({{"g", std::move(geometry)}, {"le", std::move(lagrangian)}}));
  }

  void add_dynamic_geometry(const FabDifferentialMap& forward_map,
                            const FabDynamicDifferentialMap& dynamic_map,
                            const FabDifferentialMap& geometry_map, FabLagrangian lagrangian,
                            FabGeometry geometry) {
    const auto weighted_geometry = FabWeightedGeometry(
        {{"g", std::move(geometry)}, {"le", std::move(lagrangian)}, {"ref_names", dynamic_map.ref_names()}});

    const auto pwg1 = weighted_geometry.pull(geometry_map);
    const auto pwg2 = pwg1.dynamic_pull(dynamic_map);
    const auto pwg3 = pwg2.pull(forward_map);
    geometry_ += pwg3;
  }

  void add_weighted_geometry(const FabDifferentialMap& forward_map, FabWeightedGeometry weighted_geometry) {
    const auto pulled_geometry = weighted_geometry.pull(forward_map);
    geometry_ += pulled_geometry;
    vars_ += pulled_geometry.vars();
  }

  void add_leaf(const FabLeaf* leaf, bool is_prime_leaf = false) {
    assert(leaf);

    if (const auto* attractor = dynamic_cast<const FabGenericAttractorLeaf*>(leaf)) {
      add_forcing_geometry(*leaf->map(), leaf->lagrangian(), attractor->geometry(), is_prime_leaf);
    } else if (const auto* dyn_attractor = dynamic_cast<const FabGenericDynamicAttractorLeaf*>(leaf)) {
      add_dynamic_forcing_geometry(*leaf->map(), *dyn_attractor->dynamic_map(), leaf->lagrangian(),
                                   leaf->geometry(), dyn_attractor->xdot_ref(), is_prime_leaf);
    } else if (const auto* _ = dynamic_cast<const FabGenericGeometryLeaf*>(leaf)) {
      add_geometry(*leaf->map(), leaf->lagrangian(), leaf->geometry().geom());
    } else if (const auto* dynamic_leaf = dynamic_cast<const FabGenericDynamicGeometryLeaf*>(leaf)) {
      add_dynamic_geometry(*leaf->map(), *dynamic_leaf->dynamic_map(), *dynamic_leaf->map(),
                           leaf->lagrangian(), leaf->geometry().geom());
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

  void add_forcing_geometry(FabDifferentialMap forward_map, FabLagrangian lagrangian,
                            const FabWeightedGeometry& geometry, bool is_prime_forcing_leaf) {
    forced_geometry_ = geometry;
    forced_geometry_ +=
        FabWeightedGeometry({{"g", geometry}, {"le", std::move(lagrangian)}}).pull(forward_map);
    if (is_prime_forcing_leaf) {
      forced_vars_ = geometry.vars();
      forced_forward_map_ = std::move(forward_map);
      vars_ += forced_geometry_.vars();
      geometry_.concretize();
      forced_geometry_.concretize(ref_sign_);
    }
  }

  void add_dynamic_forcing_geometry(const FabDifferentialMap& forward_map,
                                    const FabDynamicDifferentialMap& dynamic_map, FabLagrangian lagrangian,
                                    const FabWeightedGeometry& geometry, const CaSX& target_velocity,
                                    bool is_prime_forcing_leaf) {
    forced_geometry_ = geometry;
    const auto wg = FabWeightedGeometry({{"g", geometry}, {"le", std::move(lagrangian)}}).pull(forward_map);
    const auto pwg = wg.dynamic_pull(dynamic_map);
    const auto ppwg = pwg.pull(forward_map);
    forced_geometry_ += ppwg;
    if (is_prime_forcing_leaf) {
      forced_vars_ = geometry.vars();
      forced_forward_map_ = forward_map;
    }
    vars_ += forced_geometry_.vars();
    target_velocity_ += CaSX::mtimes(forward_map.J().T(), target_velocity);
    ref_sign_ = -1;
    geometry_.concretize();
    forced_geometry_.concretize(ref_sign_);
  }

  void set_execution_energy(FabLagrangian execution_lagrangian) {
    const auto composed_geom = FabGeometry({{"s", geometry_}});
    execution_lagrangian_ = std::move(execution_lagrangian);
    execution_geometry_ = FabWeightedGeometry({{"g", composed_geom}, {"le", execution_lagrangian}});
    execution_geometry_.concretize();

    try {
      const auto forced_geometry = FabGeometry({{"s", forced_geometry_}});

      forced_speed_controlled_geometry_ =
          FabWeightedGeometry({{"s", forced_geometry_}, {"le", execution_lagrangian}});

      forced_speed_controlled_geometry_.concretize();
    } catch (const FabParamNotFoundError& e) {
      std::cout << e.what() << std::endl;
    }
  }

  void set_speed_control() {
    const auto x_psi = forced_vars_.position_var();
    const auto dm_psi = forced_forward_map_;
    const auto ex_lag = execution_lagrangian_;
    const auto a_ex = CaSX::sym("a_ex_damper", 1);
    const auto a_le = CaSX::sym("a_le_damper", 1);
    // TODO: RECHECK
    damper_ = FabDamper(x_psi, config_.damper_beta(x_psi, a_ex, a_le), a_ex, a_le,
                        config_.damper_eta(CaSX::vertcat(CaSX::symvar(ex_lag.l()))), dm_psi, ex_lag.l());
    vars_.add_parameters({{a_ex.name(), a_ex}, {a_le.name(), a_le}});
  }

  CaSX get_forward_kinematics(const std::string& link_name, bool position_only = true) {
    return fk_->casadi(vars_.position_var(), link_name, FabVariant<int, std::string>(), CaSX::eye(4),
                       position_only);
  }

  void add_capsule_sphere_geometry(const std::string& obstacle_name, const std::string& capsule_name,
                                   const CaSX& tf_capsule_origin, double capsule_length) {
    const auto capsule_radius = 0.5 * capsule_length;
    auto tf_origin_center_0 = CaSX::eye(4);
    fab_core::set_casx2(tf_origin_center_0, 2, 3, capsule_radius);
    auto tf_origin_center_1 = CaSX::eye(4);
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
    auto tf_origin_center_0 = CaSX::eye(4);
    fab_core::set_casx2(tf_origin_center_0, 2, 3, capsule_radius);
    auto tf_origin_center_1 = CaSX::eye(4);
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
                                       const std::string& collision_link_name, CaSX fk) {
    auto geometry = FabObstacleLeaf(vars_, std::move(fk), obstacle_name, collision_link_name);
    geometry.set_geometry(config_.collision_geometry);
    geometry.set_finsler_structure(config_.collision_finsler);
    add_leaf(&geometry);
  }

  void add_dynamic_spherical_obstacle_geometry(const std::string& obstacle_name,
                                               const std::string& collision_link_name, const CaSX& fk,
                                               const CaSXDict& reference_params,
                                               int dynamic_obstacle_dimension = 3) {
    auto geometry = FabDynamicObstacleLeaf(vars_, fab_core::get_casx(fk, 0, dynamic_obstacle_dimension),
                                           obstacle_name, collision_link_name, reference_params);
    geometry.set_geometry(config_.collision_geometry);
    geometry.set_finsler_structure(config_.collision_finsler);
    add_leaf(&geometry);
  }

  void add_plane_constraint(std::string constraint_name, std::string collision_link_name, CaSX fk) {
    auto geometry = FabPlaneConstraintGeometryLeaf(vars_, std::move(constraint_name),
                                                   std::move(collision_link_name), std::move(fk));
    geometry.set_geometry(config_.geometry_plane_constraint);
    geometry.set_finsler_structure(config_.finsler_plane_constraint);
    add_leaf(&geometry);
  }

  void add_cuboid_obstacle_geometry(const std::string& obstacle_name, const std::string& collision_link_name,
                                    CaSX fk) {
    auto geometry = FabSphereCuboidLeaf(vars_, std::move(fk), obstacle_name, collision_link_name);
    geometry.set_geometry(config_.collision_geometry);
    geometry.set_finsler_structure(config_.collision_finsler);
    add_leaf(&geometry);
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
    auto fk = fk_2 - fk_1;
    if (fab_core::is_casx_sparse(fk)) {
      std::cout << std::string("Expression [") + fk.get_str() + "] for links " + collision_link_1_name +
                       "and " + collision_link_2_name + " is sparse and thus skipped."
                << std::endl;
      auto geometry =
          FabSelfCollisionLeaf(vars_, std::move(fk), collision_link_1_name, collision_link_2_name);
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

  void load_problem_configuration(FabProblemTextConfig problem_config) {
    problem_config_ = std::move(problem_config);
    for (const auto& obstacle : problem_config_.environment().obstacles()) {
      vars_.add_parameters(obstacle->sym_parameters());
    }

    set_collision_avoidance();
    set_self_collision_avoidance();
    set_joint_limits();
    if (fab_core::has_collection_element(
            std::array<std::string, 3>{"forced", "speed-controlled", "forced-energized"},
            config_.forcing_type)) {
      // set_goal_component(problem_config_.goal_composition);
    }
    if (fab_core::has_collection_element(
            std::array<std::string, 3>{"speed-controlled", "execution-energy", "forced-energized"},
            config_.forcing_type)) {
      set_execution_energy(FabExecutionLagrangian(vars_));
    }

    if (config_.forcing_type == "speed-controlled") {
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
        auto fk_augmented = CaSX::eye(4);
        fab_core::set_casx2(fk_augmented, {0, 2}, {0, 2}, fab_core::get_casx2(fk, {0, 2}, {0, 2}));
        fab_core::set_casx2(fk_augmented, {0, 2}, 3, fab_core::get_casx2(fk, {0, 2}, 2));
        fk = std::move(fk_augmented);
      }
      if (fk_size == decltype(fk_size){4, 4}) {
        const auto fk_0_3_3 = fab_core::get_casx2(fk, {0, 3}, 3);
        if (fab_core::is_casx_sparse(fk_0_3_3)) {
          std::cout << std::string("Expression ") + fk_0_3_3.get_str() + " for link " + link_name +
                           " is sparse and thus skipped."
                    << std::endl;
          continue;
        }
      } else if (fab_core::is_casx_sparse(fk)) {
        std::cout << std::string("Expression ") + fk.get_str() + " for link " + link_name +
                         " is sparse and thus skipped."
                  << std::endl;
        continue;
      }

      collision_link->set_origin(fk);
      vars_.add_parameters(collision_link->sym_parameters());
      vars_.add_parameter_values(collision_link->parameters());
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

  void set_components(std::vector<std::string> collision_link_names,
                      FabSelfCollisionPairs self_collision_pairs,
                      std::vector<std::string> collision_links_esdf, FabGoalComposition goal,
                      std::vector<FabJointLimit> limits, int number_obstacles = 1,
                      int number_dynamic_obstacles = 0, int number_obstacles_cuboid = 0,
                      int number_plane_constraints = 0, int dynamic_obstacle_dimension = 3) {
    // Obstacles
    std::vector<CaSXDict> reference_param_list;
    for (auto i = 0; i < number_dynamic_obstacles; ++i) {
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
        std::cout << std::string("Expression ") + fk.get_str() + " for link " + link_name +
                         "is sparse and so skipped"
                  << std::endl;
        continue;
      }
      for (auto i = 0; i < number_obstacles; ++i) {
        const auto obstacle_name = std::string("obst_") + std::to_string(i);
        add_spherical_obstacle_geometry(obstacle_name, link_name, fk);
      }
      for (auto i = 0; i < number_dynamic_obstacles; ++i) {
        const auto obstacle_name = std::string("obst_dynamic_") + std::to_string(i);
        add_dynamic_spherical_obstacle_geometry(obstacle_name, link_name, fk, reference_param_list[i],
                                                dynamic_obstacle_dimension);
      }
      for (auto i = 0; i < number_plane_constraints; ++i) {
        const auto constraint_name = std::string("constraint_") + std::to_string(i);
        add_plane_constraint(constraint_name, link_name, fk);
      }

      for (auto i = 0; i < number_obstacles_cuboid; ++i) {
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

    const auto execution_energy = FabExecutionLagrangian(vars_);
    set_execution_energy(execution_energy);
    if (true /*goal*/) {
      // set_goal_component(goal);
      set_speed_control();
    }
  }

  CaSX get_differential_map(const int sub_goal_index, const FabSubGoalPtr& sub_goal) {
    if (sub_goal->type == "staticJointSpaceSubGoal") {
      return fab_core::get_casx(vars_.position_var(), sub_goal->indices);
    } else if (sub_goal->type == "staticSubGoal") {
      const auto static_sub_goal = std::dynamic_pointer_cast<FabStaticSubGoal>(sub_goal);
      const auto fk_child = get_forward_kinematics(static_sub_goal->child_link_name);
      CaSX fk_parent;
      try {
        fk_parent = get_forward_kinematics(static_sub_goal->parent_link_name);
      } catch (const FabError& e) {
        fk_parent = CaSX::zeros(3);
      }
      return fab_core::get_casx(fk_child, sub_goal->indices) -
             fab_core::get_casx(fk_parent, sub_goal->indices);
    }
    throw FabError(sub_goal->type + " :Invalid sub-goal type");
  }

  void set_goal_component(const FabGoalComposition& goal) {
    // Adds default attractor
    const auto sub_goals = goal.sub_goals();
    for (auto i = 0; i < sub_goals.size(); ++i) {
      const auto& sub_goal = sub_goals[i];
      const auto fk_sub_goal = get_differential_map(i, sub_goal);
      if (fab_core::is_casx_sparse(fk_sub_goal)) {
        throw FabError(fk_sub_goal.get_str() + "must not be sparse");
      }
      std::unique_ptr<FabGenericDynamicAttractorLeaf> attractor = nullptr;
      if (fab_core::has_collection_element(std::vector<std::string>{"analyticSubGoal", "splineSubGoal"},
                                           sub_goal->type)) {
        attractor = std::make_unique<FabGenericDynamicAttractorLeaf>(
            vars_, fk_sub_goal, std::string("goal_") + std::to_string(i));
      } else {
        const std::string var_name = std::string("x_goal_") + std::to_string(i);
        vars_.add_parameter(var_name, CaSX::sym(var_name, sub_goal->dimension()));
        attractor = std::make_unique<FabGenericDynamicAttractorLeaf>(
            vars_, fk_sub_goal, std::string("goal_") + std::to_string(i));
        attractor->set_potential(config_.attractor_potential);
        attractor->set_metric(config_.attractor_metric);
        add_leaf(attractor.get(), sub_goal->is_primary_goal);
      }
    }
  }

  void concretize(const std::string& mode = "acc", float time_step = 0) {
    mode_ = mode;
    if ((mode == "vel") && (time_step == 0)) {
      throw FabError("No time step passed in velocity mode.");
    }
    geometry_.concretize();

    // xddot
    CaSX xddot;
    if (config_.forcing_type == "speed-controlled") {
      const CaSX eta = damper_.substitute_eta();
      const CaSX a_ex =
          (eta * execution_geometry_.alpha() + (1 - eta) * forced_speed_controlled_geometry_.alpha());
      const CaSX beta_subst = damper_.substitute_beta(-a_ex, -geometry_.alpha());
#if 1
      xddot =
          forced_geometry_.xddot() -
          (a_ex + beta_subst) * (geometry_.xdot() - CaSX::mtimes(forced_geometry_.Minv(), target_velocity_));
#else
      const CaSX xddot = self._forced_geometry._xddot;
#endif
    } else if (config_.forcing_type == "execution-energy") {
      std::cout << "No forcing term, using pure geometry with energization" << std::endl;
#if 1
      xddot = execution_geometry_.xddot() - execution_geometry_.alpha() * geometry_.vars().velocity_var();
#else
      const CaSX xddot = geometry_.xddot() - geometry_.alpha() * geometry_.vars_.velocity_var();
#endif
    } else if (config_.forcing_type == "forced-energized") {
      std::cout << "Using forced geometry with constant execution energy" << std::endl;
      xddot = forced_speed_controlled_geometry_.xddot() -
              forced_speed_controlled_geometry_.alpha() * geometry_.vars().velocity_var();
    } else if (config_.forcing_type == "forced") {
      std::cout << "No execution energy, using forced geometry without speed regulation" << std::endl;
      xddot = forced_geometry_.xddot() - geometry_.alpha() * geometry_.vars().velocity_var();
    } else if (config_.forcing_type == "pure-geometry") {
      xddot = geometry_.xddot();
    } else {
      throw FabError(config_.forcing_type + " :Unknown forcing type");
    }

    // CasadiFunction
    if (mode == "acc") {
      funs_ = FabCasadiFunction("funs", vars_, {{"action", xddot}});
    }
    if (mode == "vel") {
      assert(time_step > 0);
      funs_ = FabCasadiFunction("funs", vars_, {{"action", geometry_.xdot() + time_step * xddot}});
    }
  }

  /*
   * Computes action based on the states passed.
   * The variables passed are the joint states, and the goal position.
   * The action is nullified if its magnitude is very large or very small.
   */
  CaSX compute_action(const FabCasadiArgMap& kwargs) {
    auto eval = funs_.evaluate(kwargs);
    CaSX action = eval["action"];
    // Debugging
    std::cout << "a_ex: " << eval["a_ex"] << std::endl;

    std::cout << "alhpa_forced_geometry: " << eval["alpha_forced_geometry"] << std::endl;
    std::cout << "alpha_geometry: " << eval["alpha_geometry"] << std::endl;
    std::cout << "beta : " << eval["beta"] << std::endl;
    const double action_magnitude = double(CaSX::norm_1(action).scalar());
    if (action_magnitude < casadi::eps) {
      std::cout << "Fabrics: Avoiding small action with magnitude " << action_magnitude << std::endl;
      action = 0.0;
    } else if (action_magnitude > 1 / casadi::eps) {
      std::cout << "Fabrics: Avoiding large action with magnitude " << action_magnitude << std::endl;
      action = 0.0;
    }
    return action;
  }

 protected:
  int dof_ = 0;
  std::string mode_;
  std::shared_ptr<FabForwardKinematics> fk_ = nullptr;
  CaSX target_velocity_;
  FabPlannerConfig config_;
  FabProblemTextConfig problem_config_;
  FabVariables vars_;
  FabWeightedGeometry geometry_;
  FabWeightedGeometry forced_geometry_;
  FabVariables forced_vars_;
  FabDifferentialMap forced_forward_map_;
  std::map<std::string, FabLeaf> leaves_;
  FabLagrangian execution_lagrangian_;
  FabWeightedGeometry execution_geometry_;
  FabWeightedGeometry forced_speed_controlled_geometry_;
  FabDamper damper_;
  FabCasadiFunction funs_;
  int8_t ref_sign_ = 1;
};
