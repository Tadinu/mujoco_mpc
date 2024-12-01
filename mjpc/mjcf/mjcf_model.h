#pragma once

#include <mujoco/mujoco.h>

#include <string>

// mjpc
#include "mjpc/urdf_parser/include/model.h"

namespace mjpc {
struct MjcfModel : public urdf::UrdfModel {
  mjModel* model = nullptr;

  bool fromMjcfFile(const string& mjcf_path);
  bool fromMjcfStr(const string& xml_string);

private:
  void fill_data_structure();
  void init_link_tree(map<string, string>& parent_link_tree) override;
  void findRoot(const map<string, string>& parent_link_tree) override;
};
using MjcfModelPtr = std::shared_ptr<MjcfModel>;
}  // namespace mjpc
