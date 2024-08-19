#pragma once

#include <casadi/casadi.hpp>
#include <functional>
#include <random>
#include <string>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_environment.h"
#include "mjpc/planners/fabrics/include/fab_goal.h"
#include "mjpc/planners/fabrics/include/fab_robot_representation.h"

enum class FORCING_TYPE : uint8_t {
  SPEED_CONTROLLED,
  PURE_GEOMETRY,
  EXECUTION_ENERGY,
  FORCED,
  FORCED_ENERGIZED
};

struct FabConfigExprMeta {
  CaSX eval;  // Evaluated form, can be symbolic or scalar
  std::vector<std::string> var_names;
};

using FabConfigFunc =
    std::function<FabConfigExprMeta(const CaSX& x, const CaSX& xdot, const std::string& affix)>;
struct FabPlannerConfig {
  static CaSX sym_var(const std::string& var_name, const std::string& affix) {
    return CaSX::sym(var_name + (affix.empty() ? "" : ("_" + affix)));
  }
  FORCING_TYPE forcing_type = FORCING_TYPE::SPEED_CONTROLLED;

  FabConfigFunc base_energy = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    return FabConfigExprMeta{0.5 * 0.2 * CaSX::dot(xdot, xdot)};
  };

  FabConfigFunc collision_geometry = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    return FabConfigExprMeta{-0.5 / CaSX::pow(x, 5) * (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2)};
  };

  FabConfigFunc collision_finsler = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    return FabConfigExprMeta{0.1 / x * CaSX::pow(xdot, 2)};
  };

  FabConfigFunc limit_geometry = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    return FabConfigExprMeta{-0.1 / x * CaSX::pow(xdot, 2)};
  };

  FabConfigFunc limit_finsler = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    return FabConfigExprMeta{0.1 / x * (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2)};
  };

  FabConfigFunc self_collision_geometry = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    return FabConfigExprMeta{-0.5 / x * (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2)};
  };

  FabConfigFunc self_collision_finsler = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    return FabConfigExprMeta{0.1 / x * CaSX::pow(xdot, 2)};
  };

  FabConfigFunc geometry_plane_constraint = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    return FabConfigExprMeta{-0.5 / CaSX::pow(x, 5) * (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2)};
  };

  FabConfigFunc finsler_plane_constraint = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    return FabConfigExprMeta{0.1 / x * CaSX::pow(xdot, 2)};
  };

  // Directionally stretched metric
  FabConfigFunc attractor_potential = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    // alpha: scaling factor for the softmax
    static constexpr float alpha = 10.f;
    const CaSX x_norm = CaSX::norm_2(x);
    return FabConfigExprMeta{5.0 * (x_norm + (1 / alpha) * CaSX::log(1 + CaSX::exp(-2 * alpha * x_norm)))};
  };

  FabConfigFunc attractor_metric = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    static constexpr float alpha = 2.0;
    static constexpr float beta = 0.3;
    return FabConfigExprMeta{(alpha - beta) * CaSX::exp(-1 * CaSX::pow(0.75 * CaSX::norm_2(x), 2) + beta) *
                             fab_math::CASX_IDENTITY(x.size().first)};
  };

  FabConfigFunc damper_beta = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    const auto a_ex = sym_var("a_ex", affix);
    const auto a_le = sym_var("a_le", affix);
    return FabConfigExprMeta{
        0.5 * (CaSX::tanh(-0.5 * (CaSX::norm_2(x) - 0.02)) + 1) * 6.5 + 0.01 + CaSX::fmax(0, a_ex - a_le),
        {a_ex.name(), a_le.name()}};
  };

  FabConfigFunc damper_eta = [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
    return FabConfigExprMeta{0.5 * (CaSX::tanh(-0.9 * 0.5 * CaSX::dot(xdot, xdot) - 0.5) + 1)};
  };

  // [SYMBOLIC CONFIG] --
  //
  using FabPlannerConfigPtr = std::shared_ptr<FabPlannerConfig>;
  static FabPlannerConfigPtr get_symbolic_config() {
    static auto config = std::make_shared<FabPlannerConfig>(FabPlannerConfig{
        .base_energy =
            [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
              const auto base_inertia = sym_var("base_inertia", affix);
              return FabConfigExprMeta{0.5 * base_inertia * CaSX::dot(xdot, xdot), {base_inertia.name()}};
            },

        .collision_geometry =
            [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
              const auto k_geo = sym_var("k_geo", affix);
              const auto exp_geo = sym_var("exp_geo", affix);
              return FabConfigExprMeta{-k_geo / CaSX::pow(x, exp_geo) * CaSX::pow(xdot, 2),
                                       {k_geo.name(), exp_geo.name()}};
            },

        .collision_finsler =
            [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
              const auto k_fin = sym_var("k_fin", affix);
              const auto exp_fin = sym_var("exp_fin", affix);
              return FabConfigExprMeta{
                  k_fin / CaSX::pow(x, exp_fin) * (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2),
                  {k_fin.name(), exp_fin.name()}};
            },

        .limit_geometry =
            [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
              const auto k_limit_geo = sym_var("k_limit_geo", affix);
              const auto exp_limit_geo = sym_var("exp_limit_geo", affix);
              return FabConfigExprMeta{-k_limit_geo / CaSX::pow(x, exp_limit_geo) * CaSX::pow(xdot, 2),
                                       {k_limit_geo.name(), exp_limit_geo.name()}};
            },

        .limit_finsler =
            [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
              const auto k_limit_fin = sym_var("k_limit_fin", affix);
              const auto exp_limit_fin = sym_var("exp_limit_fin", affix);
              return FabConfigExprMeta{k_limit_fin / CaSX::pow(x, exp_limit_fin) *
                                           (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2),
                                       {k_limit_fin.name(), exp_limit_fin.name()}};
            },

        .self_collision_geometry =
            [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
              const auto k_self_geo = sym_var("k_self_geo", affix);
              const auto exp_self_geo = sym_var("exp_self_geo", affix);
              return FabConfigExprMeta{-k_self_geo / CaSX::pow(x, exp_self_geo) * CaSX::pow(xdot, 2),
                                       {k_self_geo.name(), exp_self_geo.name()}};
            },

        .self_collision_finsler =
            [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
              const auto k_self_fin = sym_var("k_self_fin", affix);
              const auto exp_self_fin = sym_var("exp_self_fin", affix);
              return FabConfigExprMeta{k_self_fin / CaSX::pow(x, exp_self_fin) *
                                           (-0.5 * (CaSX::sign(xdot) - 1)) * CaSX::pow(xdot, 2),
                                       {k_self_fin.name(), exp_self_fin.name()}};
            },

        .damper_beta =
            [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
              const auto alpha_b = sym_var("alpha_b", affix);
              const auto radius_shift = sym_var("radius_shift", affix);
              const auto beta_close = sym_var("beta_close", affix);
              const auto beta_distant = sym_var("beta_distant", affix);
              const auto a_ex = sym_var("a_ex", affix);
              const auto a_le = sym_var("a_le", affix);

              return FabConfigExprMeta{
                  0.5 * (CaSX::tanh(-alpha_b * (CaSX::norm_2(x) - radius_shift)) + 1) * beta_close +
                      beta_distant + CaSX::fmax(0, a_ex - a_le),
                  {alpha_b.name(), radius_shift.name(), beta_close.name(), beta_distant.name(), a_ex.name(),
                   a_le.name()}};
            },

        .damper_eta =
            [](const CaSX& x, const CaSX& xdot, const std::string& affix) {
              const auto alpha_eta = sym_var("alpha_eta", affix);
              const auto ex_lag = sym_var("ex_lag", affix);
              const auto ex_factor = sym_var("ex_factor", affix);
              return FabConfigExprMeta{0.5 * (CaSX::tanh(-alpha_eta * ex_lag * (1 - ex_factor) - 0.5) + 1),
                                       {alpha_eta.name(), ex_lag.name(), ex_factor.name()}};
            }});
    return config;
  }
};
using FabPlannerConfigPtr = FabPlannerConfig::FabPlannerConfigPtr;

struct FabJointLimitArray {
  std::vector<double> lower_limits;
  std::vector<double> upper_limits;
};

using FabJointLimit = std::array<double, 2>;

template <typename... TArgs>
class FabProblemConfig {
public:
  FabProblemConfig() = default;

  explicit FabProblemConfig(FabNamedMap<TArgs...> configs)
      : configs_(std::move(configs)),
        goal_composition_(
            FabGoalComposition("goal", fab_core::get_variant_value<FabGoalConfig>(configs_["goal"]))) {
    const auto& joint_limits_data = configs_["joint_limits"];
    joint_limits_ = FabJointLimitArray{
        .lower_limits = fab_core::tokenize<double>(joint_limits_data["lower_limits"], ' '),
        .upper_limits = fab_core::tokenize<double>(joint_limits_data["upper_limits"], ' ')};
    construct_robot_representation();
    const auto env_config =
        fab_core::get_variant_value<std::map<std::string, std::string>>(configs_["environment"]);
    environment_ =
        FabEnvironment(std::stoi(env_config["number_spheres"]), std::stoi(env_config["number_planes"]),
                       std::stoi(env_config["number_cuboids"]));
  }

  FabEnvironment environment() const { return environment_; }

  void construct_robot_representation() {
    std::map<std::string, FabGeometricPrimitivePtr> collision_links;
    auto robot_config = fab_core::get_variant_value<
        std::map<std::string, std::map<std::string, std::map<std::string, std::string>>>>(
        configs_["robot_representation"]);
    for (const auto& [link_name, link_data] : robot_config["collision_links"]) {
      const std::string collision_link_type = fab_core::get_map_keys(link_data)[0];
      const auto link_props = link_data[collision_link_type];

      if (collision_link_type == "capsule") {
        collision_links[link_name] = std::make_shared<FabCapsule>(link_name, std::stod(link_props["radius"]),
                                                                  std::stod(link_props["length"]));
      } else if (collision_link_type == "sphere") {
        collision_links[link_name] = std::make_shared<FabSphere>(link_name, std::stod(link_props["radius"]));
      } else if (collision_link_type == "cuboid") {
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

using FabPlannerProblemConfig =
    FabProblemConfig<FabGoalConfig, std::string, std::vector<std::string>, std::map<std::string, std::string>,
                     std::map<std::string, std::vector<std::string>>>;
