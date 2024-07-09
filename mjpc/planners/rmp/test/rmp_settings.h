#ifndef RMP_PLANNER_SETTINGS_H
#define RMP_PLANNER_SETTINGS_H
#include <Eigen/Dense>

namespace rmp {

enum PlannerType { RMP };

typedef struct TestSettings {
  int obstacles = 20;
  int seed = -1;
  int n_runs = 1;
  std::string data_path;
  std::string world_save_path;
  std::string world_load_path;
  int stats_only = 0;
} TestSettings;

/** General (3d) settings */
struct WorldGenSettings {
  explicit WorldGenSettings(TestSettings settings) {
    this->seed = settings.seed;
  };

  int seed = 0;

  std::pair<Eigen::Vector3f, Eigen::Vector3f> world_limits = {
      {0.0f, 0.0f, 0.0f}, {10.4f, 10.4f, 10.4f}};
};

}  // namespace rmp

#endif  // RMP_PLANNER_SETTINGS_H
