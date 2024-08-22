#pragma once

#include <algorithm>
#include <casadi/casadi.hpp>
#include <stdexcept>
#include <variant>

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_core_util.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"

using FabCasadiArg =
    FabVariant<CaSX, int, double, std::string, std::vector<int>, std::vector<double>, std::map<int, double>>;
using FabCasadiArgMap = std::map<std::string, FabCasadiArg>;

class FabCasadiFunction {
public:
  FabCasadiFunction() = default;

  FabCasadiFunction(std::string name, const FabVariables& variables, CaSXDict expressions)
      : name_(std::move(name)),
        inputs_(variables.all_vars()),
        expressions_(std::move(expressions)),
        arguments_(fab_core::get_casx_dict(variables.parameter_values())) {
    create_function();
  }

  std::vector<std::string> input_names_;
  CaSXVector input_values_;
  std::vector<std::string> expression_names_;
  CaSXVector expression_values_;

  void create_function() {
#if 0
    // 1- Create [input_names_, input_values_] <- [inputs_]
    input_names_ = fab_core::get_map_keys(inputs_);
    std::sort(input_names_.begin(), input_names_.end());
    std::transform(input_names_.begin(), input_names_.end(), std::back_inserter(input_values_),
                   [this](auto& input_key) { return inputs_[input_key]; });

    // 2- Create [expression_names_, expression_values_] <- [expressions_]
    expression_names_ = fab_core::get_map_keys(expressions_);
    std::sort(expression_names_.begin(), expression_names_.end());
    std::transform(expression_names_.begin(), expression_names_.end(), std::back_inserter(expression_values_),
                   [this](auto& exp_name) { return expressions_[exp_name]; });
#else
    // 1- Create [input_values_] <- [inputs_]
    input_names_ = fab_core::get_map_keys(inputs_);
    input_values_ = fab_core::get_map_values<CaSX>(inputs_);
    FAB_PRINTDB("INPUTS", input_names_.size(), input_values_.size());

    // 2- Create [expression_values_] <- [expressions_]
    expression_names_ = fab_core::get_map_keys(expressions_);
    expression_values_ = fab_core::get_map_values<CaSX>(expressions_);
    FAB_PRINTDB("EXPRESSIONS", expression_names_.size(), expression_values_.size(),
                fab_core::join(expression_names_));
#endif

    // 3- Create [function_]
    FAB_PRINTDB("CREATE FUNCTION", input_values_.size(), expression_values_.size());
    FAB_PRINTDB("INPUT VALUES", fab_core::join(input_values_));
    FAB_PRINTDB("EXPRESSION NAMES", fab_core::join(expression_names_));
    print_self();
    function_ = CaFunction(name_, input_values_, expression_values_, input_names_, expression_names_
                           /*, {{"allow_free", true}}*/);
  }

  CaFunction function() const { return function_; }

  void print_self() const {
    FAB_PRINT("Func name:", name_);
    FAB_PRINT("Input names: ", input_names_);
    FAB_PRINT("Input values: ", input_values_);
    FAB_PRINT("Expression names: ", expression_names_);
    // FAB_PRINT("Expression values: ", expression_values_);
    fab_core::print_named_map2<CaSX>(arguments_, "Args");
  }

  CaSXDict evaluate(const FabCasadiArgMap& kwargs) {
    // Process arguments
    FAB_PRINTDB("PRE-PROCESSED KWARGS", kwargs.size());
    fab_core::print_named_mapdb(kwargs);
    FAB_PRINTDB("----------------");
    // arguments_.clear();
    auto fill_arg = [this](const std::string& arg_name, const FabCasadiArg& arg,
                           const std::vector<std::string>& arg_prefix_name_list) {
      const bool bArg_matched =
          arg_prefix_name_list.empty() ||
          fab_core::has_collection_element_if(
              arg_prefix_name_list, [&arg_name](const auto& prefix) { return arg_name.starts_with(prefix); });
      if (bArg_matched) {
        CaSX arg_val;
        if (fab_core::variant_to_casx(arg, arg_val)) {
          arguments_.insert_or_assign(arg_name, arg_val);
        }
      }
      return bArg_matched;
    };

    for (const auto& [arg_name, arg] : kwargs) {
      if (fill_arg(arg_name, arg, {"x_obst", "x_obsts"})) {
      } else if (fill_arg(arg_name, arg, {"radius_obst", "radius_obsts"})) {
      } else if (fill_arg(arg_name, arg, {"x_obst_dynamic", "x_obsts_dynamic"})) {
      } else if (fill_arg(arg_name, arg, {"xdot_obst_dynamic", "xdot_obsts_dynamic"})) {
      } else if (fill_arg(arg_name, arg, {"xddot_obst_dynamic", "xddot_obsts_dynamic"})) {
      } else if (fill_arg(arg_name, arg, {"radius_obst_dynamic", "radius_obsts_dynamic"})) {
      } else if (fill_arg(arg_name, arg, {"x_obst_cuboid", "x_obsts_cuboid"})) {
      } else if (fill_arg(arg_name, arg, {"size_obst_cuboid", "size_obsts_cuboid"})) {
      } else if (arg_name.starts_with("radius_body") && arg_name.starts_with("links")) {
        std::vector<std::string> body_size_input_keys;
        std::copy_if(input_names_.begin(), input_names_.end(), std::back_inserter(body_size_input_keys),
                     [](auto& input_key) { return input_key.starts_with("radius_body"); });

        for (const auto& body_size_key : body_size_input_keys) {
          if (const auto* arg_values = std::get_if<std::map<int, double>>(&arg)) {
            for (const auto& [link_no, body_radius] : *arg_values) {
              if (body_size_key.find(std::to_string(link_no)) != std::string::npos) {
                arguments_.insert_or_assign(body_size_key, body_radius);
              }
            }
          }
        }
      } else {
        fill_arg(arg_name, arg, {});
      }
    }
    fab_core::print_named_map2db<CaSX>(arguments_, "POST-PROCESSED KWARGS");

    // Evaluate, invoking [function_(inputs)]
    // Example:
    // auto v1_dm = CaDM({1,2,3,4,5});
    // auto v2_dm = CaDM({5,4,3,2,1});
    // const auto &result = f_(std::vector<CaDM>{{v1_dm}, {v2_dm}});
    if (function_.is_null() || !function_.get()) {
      return {};
    }
    FAB_PRINTDB("CASADI FUNCTION GENERATED TO:", function_.generate(function_.name()));
    FAB_PRINTDB("CASADI FUNCTION INPUTS NUM:", function_.name_in().size());
    FAB_PRINTDB("CASADI FUNCTION OUTPUTS NUM:", function_.name_out().size());
    CaSXDict outputs = function_(arguments_);
    FAB_PRINTDB("OUTPUTS", outputs);
    for (auto& [name, val] : outputs) {
      const auto val_size = val.size();
      if ((val_size == decltype(val_size){1, 1}) || (val_size.second == 1)) {
        val = fab_core::get_casx2(val, {CASADI_INT_MIN, CASADI_INT_MAX}, 0);
      }
    }
    return outputs;
  }

protected:
  std::string name_;
  CaSXDict inputs_;
  CaSXDict expressions_;
  CaSXDict arguments_;
  CaFunction function_;
};

using FabCasadiFunctionPtr = std::shared_ptr<FabCasadiFunction>;