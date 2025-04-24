
#include "mjpc/mjcf/mjcf_model.h"

#include <fmt/format.h>

#include "mjpc/core/mjpc_common.h"
#include "mjpc/utils/mjpc_core_util.h"

namespace mjpc {
static constexpr bool MJCF_MODEL_DEBUG = true;
static constexpr bool MJCF_MODEL_WORLD_BODY_COVERED = false;

bool MjcfModel::FromMjcfFile(const string& mjcf_path) {
  std::array<char, 1024> error{};
  mj_deleteModel(model);
  model = mj_loadXML(mjcf_path.data(), nullptr, error.data(), error.size());
  mj_deleteData(data);
  data = mj_makeData(model);
  FillDataStructure();
  return bool(model);
}

bool MjcfModel::FromMjcfStr(const string& xml_string) {
  std::array<char, 1024> error{};
  mj_deleteSpec(spec);
  spec = mj_parseXMLString(xml_string.data(), nullptr, error.data(), error.size());
  if (!spec) {
    return false;
  }
  if (model) {
    mj_recompile(spec, nullptr, model, data);
  } else {
    // compile new model
    model = mj_compile(spec, nullptr);
    if (!model) {
      return false;
    }
  }
  FillDataStructure();
  return bool(model);
}

bool MjcfModel::ExportToMjcfFile(const std::string& export_path) const {
  // NOTE: This can work only if [mj_freeLastXML()] must have not been invoked before
  if (!mj_saveLastXML(export_path.c_str(), model, nullptr, 0)) {
    mjpc::print("MjcfModel [", name, "]:Failed being exported to", export_path);
    return false;
  }
  return true;
}

std::string MjcfModel::ExportToMjcfStr() const {
  // 1- Save to a temp XML file
  constexpr int kMaxPathLen = 1024;
  const std::string path_template =
      std::filesystem::temp_directory_path().append("tmp.XXXXXX").string();

  char export_path[kMaxPathLen];
  mju_strncpy(export_path, path_template.c_str(), path_template.size() + 1);

  const int fd = mkstemp(export_path);
  if (!ExportToMjcfFile(export_path)) {
    return {};
  }

  // 2- Reload the saved XML file into string
  std::string mjcf_str = ReadFile(export_path);
  close(fd);
  std::remove(export_path);
  return mjcf_str;
}

mjSpec* MjcfModel::ExportToSpec(mjVFS* vfs) const {
  std::array<char, 1024> error{};
  mjSpec* exported_spec = mj_parseXMLString(ExportToMjcfStr().data(), vfs, error.data(), error.size());
  if (error[0] != '\0') {
    std::cout << "[MjcfModel::ExportToSpec()]" << error.data() << std::endl;
  }
  return exported_spec;
}

mjSpec* MjcfModel::LoadToSpec(const std::string& mjcf, mjVFS* vfs) {
  std::array<char, 1024> error{};
  mjSpec* exported_spec = (absl::EndsWithIgnoreCase(mjcf, ".xml") || absl::EndsWithIgnoreCase(mjcf, ".mjcf"))
                            ? mj_parseXML(mjcf.c_str(), vfs, error.data(), error.size())
                            : mj_parseXMLString(mjcf.data(), vfs, error.data(), error.size());
  if (error[0] != '\0') {
    std::cout << "[MjcfModel::LoadToSpec()]" << error.data() << std::endl;
  }
  return exported_spec;
}

void MjcfModel::AddMjcfToVFS(const std::string& mjcf_path, const std::string& mjcf_registered_filename,
                             mjVFS* vfs) {
  const std::string xml = ReadFile(mjcf_path.c_str());
  mj_addBufferVFS(vfs, mjcf_registered_filename.c_str(), xml.c_str(), xml.size());
}

void MjcfModel::FillDataStructure() {
  if (!model) {
    return;
  }

  for (auto i = MJCF_MODEL_WORLD_BODY_COVERED ? 0 : 1; i < model->nbody; ++i) {
    // LINKS --
    //
    auto link = std::make_shared<urdf::Link>();
    if (auto* link_name = mj_id2name(model, mjOBJ_BODY, i)) {
      link->name = link_name;
    }
    link->origin = urdf::Transform{.position = urdf::Vector3(&model->body_pos[3 * i]),
                                   .rotation = urdf::Rotation(&model->body_quat[4 * i])};

    // Link Inertia
    link->inertial =
        urdf::Inertial{.origin = urdf::Transform{.position = urdf::Vector3(&model->body_ipos[3 * i]),
                                                 .rotation = urdf::Rotation(&model->body_iquat[4 * i])},
                       .mass = model->body_mass[i],
                       .ixx = model->body_inertia[3 * i],
                       .ixy = model->body_inertia[3 * i + 1],
                       .ixz = model->body_inertia[3 * i + 2]};

    // Link's Visual & Collision
    int geom_i = model->body_geomadr[i];
    for (auto g = geom_i; g < geom_i + model->body_geomnum[i]; ++g) {
      assert(g < model->ngeom);
      // Visual
      auto vis = std::make_shared<urdf::Visual>();
      if (auto* vis_name = mj_id2name(model, mjOBJ_GEOM, g)) {
        vis->name = vis_name;
      }
      vis->origin = urdf::Transform{.position = urdf::Vector3(&model->geom_pos[3 * g]),
                                    .rotation = urdf::Rotation(&model->geom_quat[4 * g])};
      auto geom_type = model->geom_type[g];

      // Visual Geometry
      auto& geom = vis->geometry;
      switch (geom_type) {
        case mjGEOM_SPHERE: {
          auto s = std::make_shared<urdf::Sphere>();
          s->radius = model->geom_size[3 * g];
          geom = std::move(s);
          break;
        }
        case mjGEOM_BOX: {
          auto b = std::make_shared<urdf::Box>();
          b->dim = urdf::Vector3(&model->geom_size[3 * g]);
          geom = std::move(b);
        }
        break;

        case mjGEOM_CYLINDER:
        case mjGEOM_CAPSULE: {
          auto b = std::make_shared<urdf::Cylinder>();
          b->radius = model->geom_size[3 * g];
          b->length = 2 * model->geom_size[3 * g + 1];
          geom = std::move(b);
          break;
        }

        case mjGEOM_MESH: {
          auto m = std::make_shared<urdf::Mesh>();
          m->filename = &model->paths[model->mesh_pathadr[0]];
          geom = std::move(m);
          break;
        }

        default:
          break;
      }

      // Collision
      auto col = std::make_shared<urdf::Collision>();
      col->origin = vis->origin;
      col->geometry = vis->geometry;

      // NOTE: Disregard material

      // Add both to [link]
      link->visuals.emplace_back(std::move(vis));
      link->collisions.emplace_back(std::move(col));
    } // End link's geoms
    link_map[link->name] = link;
    // End links

    if (!MJCF_MODEL_WORLD_BODY_COVERED && model->body_parentid[i] == 0) {
      continue;
    }

    // JOINTS --
    //
    const int jnt_adr = model->body_jntadr[i];
    const int jnt_num = model->body_jntnum[i];
    const auto* parent_name = mj_id2name(model, mjOBJ_BODY, model->body_parentid[i]);
    const std::string parent_link_name = parent_name ? parent_name : "";

    // Fixed joint
    if (jnt_num == 0) {
      auto joint = std::make_shared<urdf::Joint>();
      joint->name = parent_link_name + "_" + link->name + "_fixed";
      joint->type = urdf::JointType::FIXED;
      joint->parent_link_name = parent_link_name;
      joint->child_link_name = link->name;

      joint->parent_to_joint_transform = link->origin;

      // Link-joint name map: [parent_name_map], [child_name_map]
      InitLinkJointNameMap(joint);

      // [joint_map, joint_list]
      joint_map[joint->name] = joint;
      joint_list.emplace_back(std::move(joint));
    } else {
      for (auto jnt_id = jnt_adr; jnt_id < jnt_adr + jnt_num; ++jnt_id) {
        auto joint = std::make_shared<urdf::Joint>();
        if (auto* jnt_name = mj_id2name(model, mjOBJ_JOINT, jnt_id)) {
          joint->name = jnt_name;
        }
        const auto jnt_type = model->jnt_type[jnt_id];
        joint->type = (jnt_type == mjJNT_FREE)
                        ? urdf::JointType::FLOATING
                        : (jnt_type == mjJNT_BALL)
                        ? urdf::JointType::BALL
                        : (jnt_type == mjJNT_SLIDE)
                        ? urdf::JointType::PRISMATIC
                        : (jnt_type == mjJNT_HINGE)
                        ? urdf::JointType::REVOLUTE
                        : urdf::JointType::UNKNOWN;
        joint->parent_link_name = parent_link_name;
        joint->child_link_name = link->name;

        // joint axis
        joint->axis = urdf::Vector3(&model->jnt_axis[3 * jnt_id]);

        // joint pose
        auto jnt_local_pos = urdf::Vector3(&model->jnt_pos[3 * jnt_id]);
        joint->parent_to_joint_transform = link->origin * urdf::Transform{.position = jnt_local_pos};

        // Link-joint name map: [parent_name_map], [child_name_map]
        InitLinkJointNameMap(joint);

        // [joint_map, joint_list]
        joint_map[joint->name] = joint;
        joint_list.emplace_back(std::move(joint));
      }
    } // End joints
  }

  // Parent link tree
  InitParentLinkTree();

  // Debug
  if constexpr (MJCF_MODEL_DEBUG) {
    PrintSelf();
  }
}

void MjcfModel::InitLinkTree(map<string, string>& parent_link_tree) {
  UrdfModel::InitLinkTree(parent_link_tree);
  // NOTE: Since MuJoCo assumes fixed joint between parent-child bodies in case of no joint being defined
  // => Loop over body names again to fill in [parent_link_tree] to account for fixed joints also
  for (auto i = MJCF_MODEL_WORLD_BODY_COVERED ? 0 : 1; i < model->nbody; ++i) {
    const auto child_link_name = mj_id2name(model, mjOBJ_BODY, i);
    const auto parent_link_name = mj_id2name(model, mjOBJ_BODY, model->body_parentid[i]);
    // NOTE: This will overwrite the [parent_link_tree] earlier filled by [UrdfModel::]
    parent_link_tree[child_link_name] = parent_link_name;
  }
}

void MjcfModel::FindRoot(const map<string, string>& parent_link_tree) {
  // NOTE: DO NOT CALL [UrdfModel::] here!
  const auto root_link_name = mj_id2name(model, mjOBJ_BODY, 1);
  root_link = (root_link_name && link_map.contains(root_link_name)) ? link_map[root_link_name] : nullptr;
  if (root_link == nullptr) {
    throw MjpcError("Error! No root link found. The model does not contain a valid link tree.");
  }
}
} // namespace mjpc