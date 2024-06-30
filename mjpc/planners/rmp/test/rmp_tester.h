
#ifndef RMPCPP_PLANNER_TESTER_H
#define RMPCPP_PLANNER_TESTER_H

#include <string>

#include "mjpc/planners/rmp/include/core/rmp_parameters.h"
#include "mjpc/planners/rmp/include/planner/rmp_planner.h"
#include "mjpc/planners/rmp/test/rmp_settings.h"
#include "mjpc/planners/rmp/test/rmp_statistics.h"

struct ParametersWrapper {
  ~ParametersWrapper() = default;

  explicit ParametersWrapper(const RMPConfigs &params)
      : parametersRMP(params){};

  const RMPConfigs parametersRMP;
};

/**
 * The tester class makes it easy to generate worlds, run the planner and export
 * the results to files that can be evaluated in python for e.g. plotting.
 */
class Tester {
  static const int dim = 3;
  using Space = rmpcpp::Space<dim>;

 public:
  Tester(const ParametersWrapper &parameters,
         const rmpcpp::TestSettings &settings);

  void run();

 private:
  void runSingle(size_t run_index);
  void updateStats(int index, double map_density, double duration_s);
  void exportTrajectories(std::string path, const int i);
  void exportWorld(std::string path, const int i);
  void exportStats(std::string path);
  std::string getMapName();

  rmpcpp::RunStatistics statistics_;
  ParametersWrapper parameters_;
  rmpcpp::TestSettings settings_;
  std::unique_ptr<rmpcpp::RMPPlanner<Space>> planner_ = nullptr;
};

#endif  // RMPCPP_PLANNER_TESTER_H
