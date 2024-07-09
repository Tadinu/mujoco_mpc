#pragma once

#include <algorithm>
#include <casadi/casadi.hpp>
#include <stdexcept>
#include <variant>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"

using FabCasadiArg =
    FabVariant<CaSX, int, double, std::string, std::vector<int>, std::vector<double>, std::map<int, double>>;
using FabCasadiArgMap = std::map<std::string, FabCasadiArg>;

class FabCasadiFunction {
 public:
  FabCasadiFunction() = default;
  FabCasadiFunction(std::string name, const FabVariables& variables, CaSXDict expression)
      : name_(std::move(name)), inputs_(variables.vars()), expression_(std::move(expression)) {
    create_function();
  }

  std::vector<std::string> input_keys_;
  std::vector<std::string> expression_names_;
  CaSXVector list_expressions_;
  CaSXVector input_expressions_;
  std::map<int, float> input_sizes_;

  void create_function() {
    // 1- Create [list_expressions_] <- [expression_]
    expression_names_ = fab_core::get_map_keys(expression_);
    std::sort(expression_names_.begin(), expression_names_.end());
    std::transform(expression_names_.begin(), expression_names_.end(), std::back_inserter(list_expressions_),
                   [this](auto& exp_name) { return expression_[exp_name]; });

    // 2- Create [input_expressions_] <- [inputs_]
    input_keys_ = fab_core::get_map_keys(inputs_);
    std::sort(input_keys_.begin(), input_keys_.end());
    std::transform(input_keys_.begin(), input_keys_.end(), std::back_inserter(input_expressions_),
                   [this](auto& input_key) { return inputs_[input_key]; });

    // 3- Create [function_]
    function_ = casadi::Function(name_, input_expressions_, list_expressions_);
  }

  casadi::Function function() const { return function_; }

  CaSXDict evaluate(const FabCasadiArgMap& kwargs) {
    auto fill_arg = [this](const std::string& arg_name, const FabCasadiArg& arg_value,
                           const std::vector<std::string>& arg_name_list) {
      if ((arg_name == arg_name_list[0]) || (arg_name == arg_name_list[1])) {
        if (const auto* arg_values_ptr = std::get_if<std::vector<double>>(&arg_value)) {
          for (auto i = 0; i < arg_values_ptr->size(); ++i) {
            arguments_.insert_or_assign(std::string(arg_name_list[0]) + "_" + std::to_string(i),
                                        (*arg_values_ptr)[i]);
          }
        }
        return true;
      }
      return false;
    };

    for (const auto& [arg_name, arg_value] : kwargs) {
      if (fill_arg(arg_name, arg_value, {"x_obst", "x_obsts"})) {
      } else if (fill_arg(arg_name, arg_value, {"radius_obst", "radius_obsts"})) {
      } else if (fill_arg(arg_name, arg_value, {"x_obst_dynamic", "x_obsts_dynamic"})) {
      } else if (fill_arg(arg_name, arg_value, {"xdot_obst_dynamic", "xdot_obsts_dynamic"})) {
      } else if (fill_arg(arg_name, arg_value, {"xddot_obst_dynamic", "xddot_obsts_dynamic"})) {
      } else if (fill_arg(arg_name, arg_value, {"radius_obst_dynamic", "radius_obsts_dynamic"})) {
      } else if (fill_arg(arg_name, arg_value, {"x_obst_cuboid", "x_obsts_cuboid"})) {
      } else if (fill_arg(arg_name, arg_value, {"size_obst_cuboid", "size_obsts_cuboid"})) {
      } else if ((arg_name == "radius_body") || (arg_name == "links")) {
        std::vector<std::string> body_size_intputs;
        std::transform(
            input_keys_.begin(), input_keys_.end(), std::back_inserter(body_size_intputs),
            [](auto& input_key) { return (input_key.rfind("radius_body", 0) == 0) ? input_key : ""; });
        for (const auto& body_size_input : body_size_intputs) {
          if (const auto* arg_values = std::get_if<std::map<int, double>>(&arg_value)) {
            for (const auto& [link_nr, body_radius] : *arg_values) {
              if (body_size_input.find(std::to_string(link_nr)) != std::string::npos) {
                arguments_.insert_or_assign(body_size_input, body_radius);
              }
            }
          }
        }
      } else if (const auto* arg_value_ptr = std::get_if<double>(&arg_value)) {
        arguments_.insert_or_assign(arg_name, *arg_value_ptr);
      }
    }

    // Inputs
    std::vector<double> inputs;
    for (const auto& input_key : input_keys_) {
      if (arguments_.contains(input_key)) {
        inputs.push_back(std::get<double>(arguments_[input_key]));
      }
    }

    // Evaluate
    // Example:
    // auto v1_dm = casadi::DM({1,2,3,4,5});
    // auto v2_dm = casadi::DM({5,4,3,2,1});
    // const auto &result = f_(std::vector<casadi::DM>{{v1_dm}, {v2_dm}});
    const auto list_array_outputs = function_(casadi::DMVector{inputs});
    CaSXDict outputs;
    if constexpr (std::is_same_v<decltype(list_array_outputs), casadi::DM>) {
      outputs.insert_or_assign(fab_core::get_map_keys(expression_)[0], list_array_outputs);
      return outputs;
    }

    for (auto i = 0; i < expression_names_.size(); ++i) {
      const auto& expression_name = expression_names_[i];
      const auto& raw_output = list_array_outputs[i];
      const auto raw_output_size = raw_output.size();
      if ((raw_output_size == decltype(raw_output_size){1, 1}) || (raw_output_size.second == 1)) {
        casadi::Matrix<double> m;
        raw_output.get(m, true, casadi::Slice(0));
        outputs.insert_or_assign(expression_name, m);
      } else {
        outputs.insert_or_assign(expression_name, raw_output);
      }
    }
    return outputs;
  }

 protected:
  std::string name_;
  CaSXDict inputs_;
  CaSXDict expression_;
  FabNamedMap<double, std::vector<double>> arguments_;
  casadi::Function function_;
};