#ifndef RMP_PLANNER_PARSER_H
#define RMP_PLANNER_PARSER_H

#include "mjpc/planners/rmp/test/rmp_settings.h"
#include "mjpc/planners/rmp/test/rmp_tester.h"

class Parser {
 public:
  Parser(int argc, char* argv[]);

  bool parse();
  rmp::TestSettings getSettings();
  ParametersWrapper getParameters();

 private:
  ParametersWrapper getRMPParameters();
};

#endif  // RMP_PLANNER_PARSER_H
