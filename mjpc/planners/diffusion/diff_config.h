#pragma once

#include <string>

namespace mjpc {
struct DiffusionConfig {
  // Experiment
  int seed = 0;
  std::string output_dir = "output";
  int n_steps = 100;

  // Environment
  std::string env_name = "unitree_h1_walk";

  // Diffusion parameters
  int Nsample = 10;
  int Hsample = 16;
  int Hnode = 4;
  int Ndiffuse = 2;
  int Ndiffuse_init = 10;
  float temp_sample = 0.06;
  float horizon_diffuse_factor = 0.9;
  float traj_diffuse_factor = 0.5;

  std::string update_method = "mppi";
};
} // namespace mjpc
