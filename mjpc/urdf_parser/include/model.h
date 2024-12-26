#pragma once

#include <map>
#include <string>

#include "mjpc/urdf_parser/include/common.h"
#include "mjpc/urdf_parser/include/exception.h"
#include "mjpc/urdf_parser/include/joint.h"
#include "mjpc/urdf_parser/include/link.h"

#define URDF_MODEL_DEBUG_LOG (0)

using namespace std;

// Refs:
// https://github.com/ros/urdfdom
// https://github.com/bulletphysics/bullet3/tree/master/examples/Importers/ImportURDFDemo
// https://github.com/ORB-HD/URDF_Parser
// https://github.com/maxspahn/forwardKinematics/blob/develop/forwardkinematics/urdfFks/casadiConversion/urdfparser.py
namespace urdf {
struct UrdfModel {
  string name;

  // JOINTS
  map<string, JointPtr> joint_map;
  vector<JointPtr> joint_list;
  // TODO: TAKE FROM PARAM
  vector<JointType> actuated_joint_types = {urdf::JointType::PRISMATIC, urdf::JointType::REVOLUTE,
                                            urdf::JointType::CONTINUOUS};
  map<string, int> joint_name_map;
  vector<string> actuated_joint_names;
  void InitJointNameMap();
  void InitActuatedJointNames();

  using JointLinkNamePair = pair<string /*joint_name*/, string /*link_name*/>;
  map<string /*link_name*/, JointLinkNamePair> parent_name_map;
  map<string /*link_name*/, vector<JointLinkNamePair>> child_name_map;
  void InitLinkJointNameMap(const JointPtr& joint);

  vector<string> active_joint_names;
  void InitActiveJoints();

  JointPtr GetJoint(const string& joint_name) const;
  vector<JointPtr> GetJoints(const string& base_name, const string& endtip_name) const;
  int GetDof() const;

  // LINKS
  string base_link_name;
  vector<string> endtip_names;
  shared_ptr<Link> root_link = nullptr;
  map<string, LinkPtr> link_map;
  LinkPtr Root() const { return root_link; }

  LinkPtr GetLink(const string& link_name) const;
  vector<LinkPtr> GetLinks() const;

  std::vector<std::string> LinkNames() const {
    std::vector<std::string> names;
    std::transform(link_map.begin(), link_map.end(), std::back_inserter(names),
                   [](auto& link) { return link.first; });
    return names;
  }

  void InitParentLinkTree();
  virtual void InitLinkTree(map<string, string>& parent_link_tree);
  virtual void FindRoot(const map<string, string>& parent_link_tree);

  // MATERIALS
  map<string, MaterialPtr> material_map;
  MaterialPtr GetMaterial(const string& mat_name) const;

  void Clear() {
    name.clear();
    link_map.clear();
    joint_map.clear();
    parent_name_map.clear();
    material_map.clear();
    root_link = nullptr;
  }

  virtual void PrintSelf() const;

  bool FromUrdfStr(const string& xml_string);
  bool FromUrdfFile(const string& urdf_path);

  // base_name can be any link, not necessarily root
  vector<string> GetChain(const string& base_name, const string& endtip_name, bool joints = true,
                          bool links = true, bool fixed = true) const;
};

using UrdfModelPtr = std::shared_ptr<UrdfModel>;
} // namespace urdf
