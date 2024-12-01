#pragma once

#include <string>

namespace fab_core {
static std::string task_function_name(const std::string& task_name, bool is_goal_fixed, bool are_obst_fixed) {
  return task_name + (is_goal_fixed ? "_static" : "_dynamic") + "_goal" +
         (are_obst_fixed ? "_static" : "_dynamic") + "_obst";
}
}  // namespace fab_core
