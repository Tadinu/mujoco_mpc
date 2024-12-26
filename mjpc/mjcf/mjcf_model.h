#pragma once

#include <mujoco/mujoco.h>

#include <string>
#include <fstream>
#include <filesystem>

// mjpc
#include "mjpc/urdf_parser/include/model.h"

namespace mjpc {
struct MjcfModel : public urdf::UrdfModel {
  virtual ~MjcfModel() {
    mj_deleteData(data);
    mj_deleteModel(model);
    mj_deleteSpec(spec);
  }

  mjModel* model = nullptr;
  mjData* data = nullptr;
  mjSpec* spec = nullptr;

  bool FromMjcfFile(const string& mjcf_path);
  bool FromMjcfStr(const string& xml_string);

  static std::string ReadFile(const char* filepath) {
    std::ifstream ifs;
    ifs.open(filepath, std::ifstream::in);
    assert(!ifs.fail());
    std::ostringstream sstream;
    sstream << ifs.rdbuf();
    return sstream.str();
  }

#if 0
  // NOTE: This does not work yet, for unknown reason MuJoCo failed to recompile from a given to existing model
  bool ReloadBySpec(bool forced = false) {
    if (!forced && spec) {
      return true;
    }
    // Clear current data
    Clear();

    // Reload from xml
    return FromMjcfStr(ExportToMjcfStr());
  }
#endif

  bool ExportToMjcfFile(const std::string& export_path) const;
  std::string ExportToMjcfStr() const;

  mjSpec* ExportToSpec(mjVFS* vfs = nullptr) const;

  static mjSpec* LoadToSpec(const std::string& mjcf_path, mjVFS* vfs);
  static void AddMjcfToVFS(const std::string& mjcf_path, const std::string& mjcf_registered_filename,
                           mjVFS* vfs);

private:
  void FillDataStructure();
  void InitLinkTree(map<string, string>& parent_link_tree) override;
  void FindRoot(const map<string, string>& parent_link_tree) override;
};

using MjcfModelPtr = std::shared_ptr<MjcfModel>;
} // namespace mjpc
