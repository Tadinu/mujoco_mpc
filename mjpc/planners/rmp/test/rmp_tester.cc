#include "mjpc/planners/rmp/test/rmp_tester.h"

#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <Eigen/Core>

#include "mjpc/planners/rmp/include/planner/rmp_planner.h"
#include "mjpc/planners/rmp/test/rmp_settings.h"

Tester::Tester(const ParametersWrapper& parameters,
               const rmpcpp::TestSettings& settings)
    : parameters_(parameters),
      settings_(settings) {}

/**
 * Does a single planner run
 */
void Tester::runSingle(const size_t run_index) {
  planner_ = std::make_unique<rmpcpp::RMPPlanner<rmpcpp::Space<3>>>(parameters_.parametersRMP);
  auto starttime = std::chrono::high_resolution_clock::now();
  planner_->plan();
  auto endtime = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      endtime - starttime);
  double duration_s = double(duration.count()) / 1E6;

  updateStats(run_index, 0.1, duration_s);

  std::string success = planner_->success() ? "Success: " : "Failure: ";
  std::cout << success << double(duration.count()) / 1000.0 << "ms"
            << std::endl;
}

void Tester::exportTrajectories(std::string path, const int i) {
  path.append("trajectory");

  boost::format index = boost::format("%03d") % std::to_string(i);
  path.append(index.str());
  std::string endpoints_path = path;
  path.append(".txt");
  std::ofstream file;
  file.open(path, std::ofstream::trunc);  // clear file contents with trunc
  planner_->getTrajectory()->writeToStream(file);
  file.close();

  /** Export start and goal */
  endpoints_path.append("endpoints.txt");
  Eigen::IOFormat format(Eigen::FullPrecision, Eigen::DontAlignCols, " ", " ",
                         "", "", " ", "");
  file.open(endpoints_path, std::ofstream::trunc);
  file << "sx sy sz ex ey ez" << std::endl;
  file << planner_->GetStartPosQ().format(format)
       << planner_->GetGoalPosQ().format(format) << std::endl;
  file.close();
}

void Tester::run() {
  statistics_.reset(getMapName());

  for (size_t run_index = 0; run_index < (size_t)settings_.n_runs; run_index++) {
    runSingle(run_index);

    /** Export trajectories only if enabled */
    if (!settings_.stats_only) {
      exportTrajectories(settings_.data_path, run_index);
    }

    /** Export world only if enabled */
    if (!settings_.stats_only and !settings_.world_save_path.empty()) {
      exportWorld(settings_.world_save_path, run_index);
    }
    settings_.seed++;
  }
  exportStats(settings_.data_path);
}

/**
 * Export statistics to file
 * @param path
 */
void Tester::exportStats(std::string path) {
  std::ofstream f_stats;
  f_stats.open(path + "stats.txt",
               std::ofstream::trunc);  // clear file contents with trunc
  statistics_.writeSummary(f_stats);
  f_stats.close();

  std::ofstream f_stats_full;
  f_stats_full.open(path + "stats_full.txt");
  statistics_.writeLines(f_stats_full);
  f_stats_full.close();
}

std::string Tester::getMapName() {
  std::stringstream map_name;
  return map_name.str();
}

void Tester::updateStats(int index, double map_density, double duration_s) {
  bool success = planner_->success();
  auto trajectory = planner_->getTrajectory();
  rmpcpp::RunStatistics::Line stat_line;

  stat_line.success = success;
  stat_line.index = index;
  stat_line.world_density = map_density;
  stat_line.time_sec = duration_s;

  if (trajectory) {
    stat_line.integration_steps = trajectory->getSegmentCount();
    stat_line.integration_steps_per_sec =
        trajectory->getSegmentCount() / duration_s;
    stat_line.length = trajectory->getLength();
    stat_line.smoothness = trajectory->getSmoothness();
  }
  statistics_.add(stat_line);
}
