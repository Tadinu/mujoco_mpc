#pragma once

#include <algorithm>
#include <casadi/casadi.hpp>
#include <stdexcept>
#include <variant>

#include "mjpc/casadi/casadi_common.h"

class CasadiFunction {
protected:
  CasadiFunction() = default;

  CasadiFunction(std::string name, CaSXDict expressions, CaSXDict inputs = {}, CaSXDict arguments = {})
      : name_(std::move(name)),
        expressions_(std::move(expressions)),
        inputs_(std::move(inputs)),
        arguments_(std::move(arguments)) {}

  std::vector<std::string> input_names_;
  CaSXVector input_values_;
  std::vector<std::string> expression_names_;
  CaSXVector expression_values_;

  void create_function() {
#if 1
    // 1- Create [input_values_] <- [inputs_]
    input_names_.clear();
    input_values_.clear();
    for (const auto& [input_name, input_value] : inputs_) {
      input_names_.push_back(input_name);
      input_values_.push_back(input_value);
    }
    CASADI_PRINTDB("INPUTS", input_names_.size(), input_values_.size());

    // 2- Create [expression_values_] <- [expressions_]
    expression_names_.clear();
    expression_values_.clear();
    for (const auto& [exp_name, exp_value] : expressions_) {
      expression_names_.push_back(exp_name);
      expression_values_.push_back(exp_value);
    }
    CASADI_PRINTDB("EXPRESSIONS", expression_names_.size(), expression_values_.size(), expression_names_);

    // 3- Create [function_]
    CASADI_PRINT("CREATE FUNCTION");
    print_self();
    function_ = CaFunction(name_, input_values_, expression_values_, input_names_, expression_names_
                           /*, {{"allow_free", true}}*/);
#else
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
#endif
  }

public:
  CaFunction function() const { return function_; }

  virtual void print_self() const {
    CASADI_PRINT("Func name:", name_, input_values_.size(), expression_values_.size());
    CASADI_PRINT("Input names: ", input_names_);
    CASADI_PRINT("Input values: ", input_values_);
    CASADI_PRINT("Expression names: ", expression_names_);
    // CASADI_PRINT("Expression values: ", expression_values_);
    mjpc_casadi::print_named_map2<CaSX>(arguments_, "Args");
  }

  virtual CaSXDict evaluate(const CasadiArgMap& kwargs) {
    CASADI_PRINTDB(name_, "EVALUATING...");
    // Process arguments
    CASADI_PRINTDB("PRE-PROCESSED KWARGS", kwargs.size());
    mjpc_casadi::print_named_mapdb(kwargs);
    CASADI_PRINTDB("----------------");
    // arguments_.clear();
    auto fill_arg = [this](const std::string& arg_name, const CasadiArg& arg,
                           const std::vector<std::string>& arg_prefix_name_list) {
      const bool bArg_matched = arg_prefix_name_list.empty();
      if (bArg_matched) {
        CaSX arg_val;
        if (mjpc_casadi::variant_to_casx(arg, arg_val)) {
          arguments_.insert_or_assign(arg_name, arg_val);
        }
      }
      return bArg_matched;
    };

    for (const auto& [arg_name, arg] : kwargs) {
      fill_arg(arg_name, arg, {});
    }
    mjpc_casadi::print_named_map2db<CaSX>(arguments_, "POST-PROCESSED KWARGS");

    // Evaluate, invoking [function_(inputs)]
    // Example:
    // auto v1_dm = CaDM({1,2,3,4,5});
    // auto v2_dm = CaDM({5,4,3,2,1});
    // const auto &result = f_(std::vector<CaDM>{{v1_dm}, {v2_dm}});
    if (function_.is_null() || !function_.get()) {
      return {};
    }

    CASADI_PRINTDB("CASADI FUNCTION INPUTS NUM:", function_.name_in().size());
    CASADI_PRINTDB("CASADI FUNCTION OUTPUTS NUM:", function_.name_out().size());

    // Invoke [function_]
    const auto start = std::chrono::high_resolution_clock::now();
    CaSXDict outputs = function_(arguments_);

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    CASADI_PRINTDB("Average compute time:", double(duration.count()) * 0.001, " milliseconds");
    CASADI_PRINTDB("OUTPUTS", outputs);
    for (auto& [name, val] : outputs) {
      const auto val_size = val.size();
      if ((val_size == decltype(val_size){1, 1}) || (val_size.second == 1)) {
        val = mjpc_casadi::get_casx2(val, {CASADI_INT_MIN, CASADI_INT_MAX}, 0);
      }
    }
    return outputs;
  }

protected:
  std::string name_;
  CaSXDict expressions_;
  CaSXDict inputs_;
  CaSXDict arguments_;
  CaFunction function_;
};

using CasadiFunctionPtr = std::shared_ptr<CasadiFunction>;
