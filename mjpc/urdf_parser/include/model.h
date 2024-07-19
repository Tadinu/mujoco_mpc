#pragma once

#include <map>
#include <string>

#include "mjpc/urdf_parser/include/common.h"
#include "mjpc/urdf_parser/include/exception.h"
#include "mjpc/urdf_parser/include/joint.h"
#include "mjpc/urdf_parser/include/link.h"

#define URDF_MODEL_DEBUG_LOG (0)

using namespace std;

namespace urdf {
struct UrdfModel {
  string name;

  // JOINTS
  map<string, JointPtr> joint_map;
  vector<JointPtr> joint_list;
  vector<JointType> actuated_joint_types;
  map<string, int> joint_name_map;
  vector<string> actuated_joint_names;
  void init_joint_name_map();
  void init_actuated_joint_names();

  using JointLinkNamePair = pair<string /*joint_name*/, string /*link_name*/>;
  map<string /*link_name*/, JointLinkNamePair> parent_name_map;
  map<string /*link_name*/, vector<JointLinkNamePair>> child_name_map;

  vector<string> active_joint_names;
  void init_active_joints();

  JointPtr get_joint(const string& joint_name) const;
  vector<JointPtr> get_joints(const string& base_name, const string& endtip_name) const;
  int get_dof() const;

  // LINKS
  string base_link_name;
  vector<string> endtip_names;
  shared_ptr<Link> root_link = nullptr;
  map<string, LinkPtr> link_map;
  LinkPtr root() const { return root_link; }

  LinkPtr get_link(const string& link_name) const;
  vector<LinkPtr> get_links() const;
  std::vector<std::string> link_names() const {
    std::vector<std::string> names;
    std::transform(link_map.begin(), link_map.end(), std::back_inserter(names),
                   [](auto& link) { return link.first; });
    return names;
  }

  void init_link_tree(map<string, string>& parent_link_tree);
  void findRoot(const map<string, string>& parent_link_tree);

  // MATERIALS
  map<string, MaterialPtr> material_map;
  MaterialPtr get_material(const string& mat_name) const;

  void clear() {
    name.clear();
    link_map.clear();
    joint_map.clear();
    parent_name_map.clear();
    material_map.clear();
    root_link = nullptr;
  }

  void print_self() const;

  bool fromUrdfStr(const string& xml_string);
  bool fromUrdfFile(const string& urdf_path);

  // base_name can be any link, not necessarily root
  vector<string> get_chain(const string& base_name, const string& endtip_name, bool joints = true,
                           bool links = true, bool fixed = true) const;
};
}  // namespace urdf
