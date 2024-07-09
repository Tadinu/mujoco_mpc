#pragma once

#include <casadi/casadi.hpp>
#include <functional>
#include <string>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_environment.h"
#include "mjpc/planners/fabrics/include/fab_goal.h"
#include "mjpc/planners/fabrics/include/fab_robot_representation.h"

struct FabPlannerConfig {
  std::string forcing_type = "speed-controlled";  // options are 'speed-controlled', 'pure-geometry',
                                                  // 'execution-energy', 'forced', 'forced-energized'
  std::function<CaSX(const CaSX& xdot)> base_energy = [](const CaSX& xdot) {
    return 0.5 * 0.2 * CaSX::dot(xdot, xdot);
  };
  std::function<CaSX(const CaSX& x, const CaSX& xdot)> collision_geometry = [](const CaSX& x,
                                                                               const CaSX& xdot) {
    return -0.5 / CaSX::pow(x, 5) * (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2);
  };

  std::function<CaSX(const CaSX& x, const CaSX& xdot)> collision_finsler =
      [](const CaSX& x, const CaSX& xdot) { return 0.1 / x * CaSX::pow(xdot, 2); };

  std::function<CaSX(const CaSX& x, const CaSX& xdot)> limit_geometry = [](const CaSX& x, const CaSX& xdot) {
    return -0.1 / x * CaSX::pow(xdot, 2);
  };

  std::function<CaSX(const CaSX& x, const CaSX& xdot)> limit_finsler = [](const CaSX& x, const CaSX& xdot) {
    return 0.1 / x * (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2);
  };

  std::function<CaSX(const CaSX& x, const CaSX& xdot)> self_collision_geometry = [](const CaSX& x,
                                                                                    const CaSX& xdot) {
    return -0.5 / x * (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2);
  };

  std::function<CaSX(const CaSX& x, const CaSX& xdot)> self_collision_finsler =
      [](const CaSX& x, const CaSX& xdot) { return 0.1 / x * CaSX::pow(xdot, 2); };

  std::function<CaSX(const CaSX& x, const CaSX& xdot)> geometry_plane_constraint = [](const CaSX& x,
                                                                                      const CaSX& xdot) {
    return -0.5 / CaSX::pow(x, 5) * (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2);
  };

  std::function<CaSX(const CaSX& x, const CaSX& xdot)> finsler_plane_constraint =
      [](const CaSX& x, const CaSX& xdot) { return 0.1 / x * CaSX::pow(xdot, 2); };

  std::function<CaSX(const CaSX& x)> attractor_potential = [](const CaSX& x) {
    return (5.0 / CaSX::norm_2(x)) + (1 / (10 * CaSX::log(1 + CaSX::exp(-2 * 10 * CaSX::norm_2(x)))));
  };

  std::function<CaSX(const CaSX& x)> attractor_metric = [](const CaSX& x) {
    return ((2.0 - 0.3) * CaSX::exp(-1 * CaSX::pow(0.75 * CaSX::norm_2(x), 2) + 0.3) *
            CaSX::eye(x.size().first));
  };

  std::function<CaSX(const CaSX& x, const CaSX& a_ex, const CaSX& a_le)> damper_beta = [](const CaSX& x,
                                                                                          const CaSX& a_ex,
                                                                                          const CaSX& a_le) {
    return 0.5 * (CaSX::tanh(-0.5 * (CaSX::norm_2(x) - 0.02)) + 1) * 6.5 + 0.01 + CaSX::fmax(0, a_ex - a_le);
  };

  std::function<CaSX(const CaSX& xdot)> damper_eta = [](const CaSX& xdot) {
    return 0.5 * (CaSX::tanh(-0.9 * 0.5 * CaSX::dot(xdot, xdot) - 0.5) + 1);
  };

  std::function<CaSX(const CaSX& x, const CaSX& a_ex, const CaSX& a_le, const CaSX& alpha_b,
                     const CaSX& radius_shift, const CaSX& beta_close, const CaSX& beta_distant)>
      damper_beta_ = [](const CaSX& x, const CaSX& a_ex, const CaSX& a_le, const CaSX& alpha_b,
                        const CaSX& radius_shift, const CaSX& beta_close, const CaSX& beta_distant) {
        return 0.5 * (CaSX::tanh(-alpha_b * (CaSX::norm_2(x) - radius_shift)) + 1) * beta_close +
               beta_distant + CaSX::fmax(0, a_ex - a_le);
      };

  std::function<CaSX(const CaSX& alpha_eta, const CaSX& ex_lag, const CaSX& ex_factor)> damper_eta_ =
      [](const CaSX& alpha_eta, const CaSX& ex_lag, const CaSX& ex_factor) {
        return 0.5 * (CaSX::tanh(-alpha_eta * ex_lag * (1 - ex_factor) - 0.5) + 1);
      };
};

struct FabSubgoal {
  std::string child_link;
  std::vector<double> desired_position;
  double epsilon = 0.;
  std::vector<int> indices;
  bool is_primary_goal = false;
  std::string parent_link;
  std::string type;
  double weight = 0.;
};

struct FabJointLimitArray {
  std::vector<double> lower_limits;
  std::vector<double> upper_limits;
};
using FabJointLimit = std::array<double, 2>;

template <typename... TArgs>
class FabProblemConfig {
 public:
  FabProblemConfig() = default;

  explicit FabProblemConfig(FabNamedMap<TArgs...> configs) : configs_(std::move(configs)) {
    // self._goal_composition=GoalComposition(name='goal',
    // content_dict=self._config['goal']['goal_definition'])
    const auto& joint_limits_data = configs_["joint_limits"];
    joint_limits_ = FabJointLimitArray{
        .lower_limits = fab_core::tokenize<double>(joint_limits_data["lower_limits"], ' '),
        .upper_limits = fab_core::tokenize<double>(joint_limits_data["upper_limits"], ' ')};
    construct_robot_representation();
    const auto env_config = std::get<std::map<std::string, std::string>>(configs_["environment"]);
    environment_ =
        FabEnvironment(std::stoi(env_config["number_spheres"]), std::stoi(env_config["number_planes"]),
                       std::stoi(env_config["number_cuboids"]));
  }

  FabEnvironment environment() const { return environment_; }
  void construct_robot_representation() {
    std::map<std::string, FabGeometricPrimitivePtr> collision_links;
    auto robot_config =
        std::get<std::map<std::string, std::map<std::string, std::map<std::string, std::string>>>>(
            configs_["robot_representation"]);
    for (const auto& [link_name, link_data] : robot_config["collision_links"]) {
      const std::string collision_link_type = fab_core::get_map_keys(link_data)[0];
      const auto link_props = link_data[collision_link_type];

      if (collision_link_type == "Capsule") {
        collision_links[link_name] = std::make_shared<FabCapsule>(link_name, std::stod(link_props["radius"]),
                                                                  std::stod(link_props["length"]));
      } else if (collision_link_type == "Sphere") {
        collision_links[link_name] = std::make_shared<FabSphere>(link_name, std::stod(link_props["radius"]));
      } else if (collision_link_type == "Cuboid") {
        auto sizes = fab_core::tokenize<double>(link_props["size"], ' ');
        assert(sizes.size() == 3);
        collision_links[link_name] = std::make_shared<FabCuboid>(link_name, std::move(sizes));
      }
    }
    robot_representation_ =
        FabRobotRepresentation(std::move(collision_links), std::move(robot_config["self_collision_pairs"]));
  }
  FabGoalComposition goal_composition() const { return goal_composition_; }

  FabRobotRepresentation robot_representation() const { return robot_representation_; }

  FabJointLimitArray joint_limits() const { return joint_limits_; }

 protected:
  FabJointLimitArray joint_limits_;
  FabNamedMap<TArgs...> configs_;
  FabEnvironment environment_;
  FabGoalComposition goal_composition_;
  FabRobotRepresentation robot_representation_;
};

using FabProblemTextConfig =
    FabProblemConfig<std::string, std::vector<std::string>, std::map<std::string, std::string>,
                     std::map<std::string, std::vector<std::string>>>;
