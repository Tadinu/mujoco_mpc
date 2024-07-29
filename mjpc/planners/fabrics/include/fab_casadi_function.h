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

  FabCasadiFunction(std::string name, const FabVariables& variables, CaSXDict expressions)
    : name_(std::move(name)), inputs_(variables.vars()), expressions_(std::move(expressions)) {
    create_function();
  }

  std::vector<std::string> input_keys_;
  std::vector<std::string> expression_names_;
  CaSXVector list_expressions_;
  CaSXVector input_expressions_;
  std::map<int, float> input_sizes_;

  void create_function() {
    // 1- Create [list_expressions_] <- [expression_]
    expression_names_ = fab_core::get_map_keys(expressions_);
    std::sort(expression_names_.begin(), expression_names_.end());
    std::transform(expression_names_.begin(), expression_names_.end(), std::back_inserter(list_expressions_),
                   [this](auto& exp_name) { return expressions_[exp_name]; });

    // 2- Create [input_expressions_] <- [inputs_]
    input_keys_ = fab_core::get_map_keys(inputs_);
    std::sort(input_keys_.begin(), input_keys_.end());
    std::transform(input_keys_.begin(), input_keys_.end(), std::back_inserter(input_expressions_),
                   [this](auto& input_key) { return inputs_[input_key]; });

    // 3- Create [function_]
    FAB_PRINTDB(fab_core::join(input_expressions_), ",\n");
    FAB_PRINTDB("CREATE FUNCTION", input_expressions_.size(), list_expressions_.size());
    function_ = CaFunction(name_, input_expressions_, list_expressions_ /*, {{"allow_free", true}}*/);
  }

  CaFunction function() const { return function_; }

  void print_self() const {
    FAB_PRINT(name_);
    FAB_PRINT(input_expressions_);
    FAB_PRINT(list_expressions_);
  }

  CaSXDict evaluate(const FabCasadiArgMap& kwargs) {
    FAB_PRINTDB("KWARGS", kwargs.size());
    fab_core::print_named_mapdb(kwargs);
    FAB_PRINTDB("----------------");
    arguments_.clear();
    auto fill_arg = [this](const std::string& arg_name, const FabCasadiArg& arg_value,
                           const std::vector<std::string>& arg_prefix_name_list) {
      const bool bArg_matched =
          arg_prefix_name_list.empty() ||
          fab_core::has_collection_element_if(arg_prefix_name_list, [&arg_name](const auto& prefix) {
            return arg_name.starts_with(prefix);
          });
      if (bArg_matched) {
        arguments_.insert_or_assign(arg_name, fab_core::get_variant_value_any(arg_value));
      }
      return bArg_matched;
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
        std::vector<std::string> body_size_input_keys;
        std::copy_if(input_keys_.begin(), input_keys_.end(), std::back_inserter(body_size_input_keys),
                     [](auto& input_key) { return absl::StartsWith(input_key, "radius_body"); });

        for (const auto& body_size_key : body_size_input_keys) {
          if (const auto* arg_values = std::get_if<std::map<int, double>>(&arg_value)) {
            for (const auto& [link_no, body_radius] : *arg_values) {
              if (body_size_key.find(std::to_string(link_no)) != std::string::npos) {
                arguments_.insert_or_assign(body_size_key, body_radius);
              }
            }
          }
        }
      } else {
        fill_arg(arg_name, arg_value, {});
      }
    }
    FAB_PRINTDB("ARGUMENTS", arguments_.size());
    FAB_PRINTDB("----------------");
    FAB_PRINTDB("INPUTS");

    // Inputs
    CaSXVector inputs;
    for (const auto& input_key : input_keys_) {
      if (arguments_.contains(input_key)) {
        const auto arg_any = arguments_[input_key];
        if (arg_any.has_value()) {
#define GET_ANY_VAL(TVal, any_val)                     \
  try {                                                \
    inputs.emplace_back(std::any_cast<TVal>(any_val)); \
  } catch (const std::bad_any_cast& e) {               \
  }
          GET_ANY_VAL(int, arg_any);
          GET_ANY_VAL(double, arg_any);
          GET_ANY_VAL(std::vector<int>, arg_any);
          GET_ANY_VAL(std::vector<double>, arg_any);
          GET_ANY_VAL(CaSX, arg_any);
          // GET_ANY_VAL(std::vector<std::map<int, double>>, arg_any);
          // GET_ANY_VAL(std::vector<std::string>, arg_any);
          FAB_PRINTDB(input_key, ": ", inputs.back());
        }
      }
    }

    // Evaluate
    // Example:
    // auto v1_dm = casadi::DM({1,2,3,4,5});
    // auto v2_dm = casadi::DM({5,4,3,2,1});
    // const auto &result = f_(std::vector<casadi::DM>{{v1_dm}, {v2_dm}});
    FAB_PRINTDB("INPUTS SIZE", inputs.size());
    FAB_PRINTDB("----------------");
    const CaSXVector array_outputs = function_(inputs);
    FAB_PRINTDB("ARRAY OUTPUTS", array_outputs);
    CaSXDict outputs;
    if constexpr (std::is_same_v<decltype(array_outputs), casadi::DM>) {
      outputs.insert_or_assign(
          fab_core::get_map_keys(expressions_)[0],
          fab_core::get_casx2(
              array_outputs, {std::numeric_limits<casadi_int>::min(), std::numeric_limits<casadi_int>::max()},
              0));
      return outputs;
    }

    for (auto i = 0; i < expression_names_.size(); ++i) {
      const auto& expression_name = expression_names_[i];
      const auto& raw_output = array_outputs[i];
      const auto raw_output_size = raw_output.size();
      if ((raw_output_size == decltype(raw_output_size){1, 1}) || (raw_output_size.second == 1)) {
        outputs.insert_or_assign(
            expression_name,
            fab_core::get_casx2(
                raw_output, {std::numeric_limits<casadi_int>::min(), std::numeric_limits<casadi_int>::max()},
                0));
      } else {
        outputs.insert_or_assign(expression_name, raw_output);
      }
    }
    return outputs;
  }

protected:
  std::string name_;
  CaSXDict inputs_;
  CaSXDict expressions_;
  std::map<std::string, std::any> arguments_;
  CaFunction function_;
};

using FabCasadiFunctionPtr = std::shared_ptr<FabCasadiFunction>;