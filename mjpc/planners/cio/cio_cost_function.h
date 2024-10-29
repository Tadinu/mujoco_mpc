#pragma once

#include <algorithm>
#include <casadi/casadi.hpp>
#include <stdexcept>
#include <variant>

#include "mjpc/casadi/casadi_common.h"
#include "mjpc/casadi/casadi_function.h"

class CIOCostFunction : public CasadiFunction {
public:
  CIOCostFunction() = default;

  CIOCostFunction(std::string name, CaSXDict expressions, CaSXDict inputs = {}, CaSXDict arguments = {})
      : CasadiFunction(std::move(name), std::move(expressions), std::move(inputs), std::move(arguments)) {
    create_function();
  }
};

using CIOCostFunctionPtr = std::shared_ptr<CIOCostFunction>;
