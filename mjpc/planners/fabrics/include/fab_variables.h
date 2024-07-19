#pragma once

#include <any>
#include <casadi/casadi.hpp>
#include <stdexcept>
#include <variant>

#include "mjpc/planners/fabrics/include/fab_core_util.h"

class FabVariables {
 public:
  using FabTrajectory = FabVariables;
  using FabTrajectories = std::vector<FabTrajectory>;
  FabVariables() = default;

  explicit FabVariables(CaSXDict state_variables, CaSXDict parameters = CaSXDict(),
                        FabDoubleScalarMap parameter_values = FabDoubleScalarMap())
      : state_variables_(std::move(state_variables)),
        parameters_(std::move(parameters)),
        parameter_values_(std::move(parameter_values)) {}

  void print_self() const {
    const auto self_size = size();
    const auto vars_size = vars().size();
    FAB_PRINT("SIZE", self_size, "VARS SIZE", vars_size);
    assert(self_size == vars_size);
    FAB_PRINT("STATE VARS --");
    for (const auto& [var_name, var] : state_variables_) {
      FAB_PRINT(var_name, ":", var);
    }

    FAB_PRINT("PARAMS --");
    for (const auto& [param_name, param] : parameters_) {
      FAB_PRINT(param_name, ":", param);
    }

    FAB_PRINT("PARAM VALS --");
    for (const auto& [param_val_name, param_val] : parameter_values_) {
      FAB_PRINT(param_val_name, ":");
      fab_core::print_variant(param_val);
    }
  }

  CaSX position_var() const {
    const auto state_variable_names = fab_core::get_map_keys(state_variables_);
    if (state_variable_names.empty()) {
      throw FabParamNotFoundError("There is 0 state variable");
    }
    return state_var(state_variable_names[0]);
  }

  CaSX velocity_var() const {
    const auto state_variable_names = fab_core::get_map_keys(state_variables_);
    if (state_variable_names.size() < 2) {
      throw FabParamNotFoundError("There are <2 state variables");
    }
    return state_var(state_variable_names[1]);
  }

  CaSXDict vars() const {
    CaSXDict all;
    append_variants<CaSX>(all, state_variables_, true);
    append_variants<CaSX>(all, parameters_, true);
    return all;
  }

  size_t size() const { return state_variables_.size() + parameters_.size(); }
  bool empty() const { return state_variables_.empty() && parameters_.empty(); }

  CaSX state_var(const std::string& name) const {
    return state_variables_.contains(name) ? state_variables_.at(name) : CaSX();
  }

  CaSXDict state_variables() const { return state_variables_; }

  void add_state_variable(std::string name, CaSX value) {
    state_variables_.insert_or_assign(std::move(name), std::move(value));
  }

  CaSXDict parameters() const { return parameters_; }

  CaSX parameter(const std::string& name) const {
    return parameters_.contains(name) ? parameters_.at(name) : CaSX();
  }

  double parameter_value(const std::string& name) {
    return parameter_values_.contains(name) ? fab_core::get_variant_value<double>(parameter_values_.at(name))
                                            : -1;
  }

  void add_parameter(std::string name, CaSX value) {
    parameters_.insert_or_assign(std::move(name), std::move(value));
  }

  void add_parameters(CaSXDict params) {
    for (auto param : std::move(params)) {
      parameters_.insert_or_assign(std::move(param.first), std::move(param.second));
    }
  }

  FabDoubleScalarMap parameter_values() const { return parameter_values_; }

  void add_parameter_values(FabDoubleScalarMap param_values) {
    for (auto param_value : std::move(param_values)) {
      parameter_values_.insert_or_assign(std::move(param_value.first), std::move(param_value.second));
    }
  }

  template <typename T, typename TNamedMap = std::map<std::string, T>>
  static void append_variants(TNamedMap& variants1, const TNamedMap& variants2, bool overwrite = false) {
    for (const auto& other_var : variants2) {
      const auto& other_key = other_var.first;
      const auto& other_val = other_var.second;
      if (overwrite) {
        variants1.insert_or_assign(other_key, other_val);
      } else if constexpr (std::is_same_v<T, CaSX>) {
        if (variants1.contains(other_key)) {
          if (!CaSX::is_equal(variants1[other_key], other_val)) {
            std::string new_key = other_key;
            int counter = 0;
            do {
              new_key.append('_' + std::to_string(counter++));
            } while (variants1.contains(new_key));
            variants1[new_key] = other_val;
          }
        } else {
          variants1.insert_or_assign(other_key, other_val);
        }
      }
    }
  }

  FabVariables operator+(const FabVariables& other) const {
    FabVariables v = *this;
    v += other;
    return v;
  }

  FabVariables& operator+=(const FabVariables& other) {
    append_variants<CaSX>(state_variables_, other.state_variables());
    append_variants<CaSX>(parameters_, other.parameters());
    append_variants<std::variant<double>>(parameter_values_, other.parameter_values(), true);
    return *this;
  }

  bool operator==(const FabVariables& other) const {
    return fab_core::is_equal_SXDict(state_variables_, other.state_variables_) &&
           fab_core::is_equal_SXDict(parameters_, other.parameters_) &&
           fab_core::is_equal_itertable(parameter_values_, other.parameter_values_);
  }

  static FabTrajectories join_refTrajs(const FabTrajectories& refTrajs1, const FabTrajectories& refTrajs2) {
    FabTrajectories refTrajs = refTrajs1;
    std::copy(refTrajs2.begin(), refTrajs2.end(), std::back_inserter(refTrajs));
    FabTrajectories unique_items;
    for (const auto& item : refTrajs) {
      bool already_exists = false;
      for (const auto& u_item : unique_items) {
        if (u_item == item) {
          already_exists = true;
          break;
        }
      }
      if (!already_exists) {
        unique_items.push_back(item);
      }
    }
    return unique_items;
  }

 protected:
  CaSXDict state_variables_;
  CaSXDict parameters_;
  FabDoubleScalarMap parameter_values_;
};

using FabTrajectory = FabVariables::FabTrajectory;
using FabTrajectories = FabVariables::FabTrajectories;

namespace ref {
// dx/dt = f(x,u)
static casadi::MX f(const casadi::MX& x, const casadi::MX& u) { return vertcat(x(1), u - x(1)); }

static int opt_race_car() {
  // Car race along a track
  // ----------------------
  // An optimal control problem (OCP), solved with direct multiple-shooting.
  // For more information see: http://labs.casadi.org/OCP
  int N = 100;                         // number of control intervals
  casadi::Opti opti = casadi::Opti();  // Optimization problem

  casadi::Slice all;
  // ---- decision variables ---------
  casadi::MX X = opti.variable(2, N + 1);  // state trajectory
  auto pos = X(0, all);
  auto speed = X(1, all);
  casadi::MX U = opti.variable(1, N);  // control trajectory (throttle)
  casadi::MX T = opti.variable();      // final time

  // ---- objective          ---------
  opti.minimize(T);  // race in minimal time

  // ---- dynamic constraints --------
  casadi::MX dt = T / N;
  for (int k = 0; k < N; ++k) {
    casadi::MX k1 = f(X(all, k), U(all, k));
    casadi::MX k2 = f(X(all, k) + dt / 2 * k1, U(all, k));
    casadi::MX k3 = f(X(all, k) + dt / 2 * k2, U(all, k));
    casadi::MX k4 = f(X(all, k) + dt * k3, U(all, k));
    casadi::MX x_next = X(all, k) + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
    opti.subject_to(X(all, k + 1) == x_next);  // close the gaps
  }

  // ---- path constraints -----------
  opti.subject_to(speed <= 1 - sin(2 * casadi::pi * pos) / 2);  // track speed limit
  opti.subject_to(0 <= U <= 1);                                 // control is limited

  // ---- boundary conditions --------
  opti.subject_to(pos(0) == 0);    // start at position 0 ...
  opti.subject_to(speed(0) == 0);  // ... from stand-still
  opti.subject_to(pos(N) == 1);    // finish line at position 1

  // ---- misc. constraints  ----------
  opti.subject_to(T >= 0);  // Time must be positive

  // ---- initial values for solver ---
  opti.set_initial(speed, 1);
  opti.set_initial(T, 1);

  // ---- solve NLP              ------
  opti.solver("ipopt");                // set numerical backend
  casadi::OptiSol sol = opti.solve();  // actual solve
  return 0;
}
}  // namespace ref

#if 0
class Variables(object):
    def __init__(self, state_variables=None, parameters=None, parameters_values=None):
        if state_variables is None:
            state_variables = {}
        if parameters is None:
            parameters = {}
        if parameters_values is None:
            parameters_values = {}
        self._state_variables = state_variables
        self._parameters = parameters
        self._parameters_values = parameters_values
        if len(state_variables) > 0:
            self._state_variable_names = list(state_variables.keys())
        if len(parameters) > 0:
            self._parameter_names = list(parameters.keys())

    def state_variables(self):
        return self._state_variables

    def add_state_variable(self, name, value):
        self._state_variables[name] = value

    def parameters(self) -> dict:
        return self._parameters

    def parameters_values(self) -> dict:
        return self._parameters_values

    def add_parameter(self, name: str, value: ca.SX) -> None:
        self._parameters[name] = value

    def add_parameter_value(self, name: str, value: Union[float, np.ndarray]) -> None:
        if name not in self._parameters:
            raise FabParamNotFoundError(f"Parameter {name} not in parameters")
        self._parameters_values[name] = value

    def add_parameters(self, parameter_dict: dict) -> None:
        self._parameters.update(parameter_dict)

    def add_parameters_values(self, parameter_dict: dict) -> None:
        for parameter_name, parameter_value in parameter_dict.items():
            self.add_parameter_value(parameter_name, parameter_value)

    def set_parameters(self, parameters):
        self._parameters = parameters

    def variable_by_name(self, name: str) -> ca.SX:
        return self._state_variables[name]

    def parameter_by_name(self, name: str) -> ca.SX:
        try:
            return self._parameters[name]
        except KeyError as key_error:
            raise FabParamNotFoundError(f"Parameter {name} not in variables, available ones are {self._parameters.keys()}")

    def position_variable(self) -> ca.SX:
        return self.variable_by_name(self._state_variable_names[0])

    def velocity_variable(self) -> ca.SX:
        return self.variable_by_name(self._state_variable_names[1])

    def verify(self):
        for key in self._state_variables:
            assert isinstance(key, str)
            assert isinstance(self._state_variables[key], ca.SX)
        for key in self._parameters:
            assert isinstance(key, str)
            assert isinstance(self._parameters[key], ca.SX)

    def asDict(self):
        joinedDict = {}
        joinedDict.update(self._state_variables)
        joinedDict.update(self._parameters)
        return joinedDict

    def __add__(self, b):
        joined_state_variables = deepcopy(self._state_variables)
        for key, value in b.state_variables().items():
            if key in joined_state_variables:
                if ca.is_equal(joined_state_variables[key], value):
                    continue
                else:
                    new_key = key
                    counter = 1
                    while new_key in joined_state_variables.keys():
                        new_key = key + "_" + str(counter)
                        counter += 1
                    joined_state_variables[new_key] = value
            else:
                joined_state_variables[key] = value
        joined_parameters = deepcopy(self._parameters)
        for key, value in b.parameters().items():
            if key in joined_parameters:
                if ca.is_equal(joined_parameters[key], value):
                    continue
                else:
                    new_key = key
                    counter = 1
                    while new_key in joined_parameters.keys():
                        new_key = key + "_" + str(counter)
                        counter += 1
                    joined_parameters[new_key] = value
            else:
                joined_parameters[key] = value
        joined_parameters_values = {**self.parameters_values(), **b.parameters_values()}
        return Variables(
            state_variables=joined_state_variables,
            parameters=joined_parameters,
            parameters_values=joined_parameters_values,
        )

    def len(self):
        return len(self._parameters.values()) + len(
            self._state_variables.values()
        )

    def __repr__(self):
        return self.__str__();

    def __str__(self):
        return (
            "State variables: "
            + self._state_variables.__str__()
            + "| parameters : "
            + self._parameters.__str__()
        )
#endif