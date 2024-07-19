#include "mjpc/urdf_parser/include/model.h"

#include <fstream>

// abseil
#include "absl/strings/ascii.h"
#include "absl/types/any.h"
#include "absl/types/bad_any_cast.h"

// urdf_parser
#include "mjpc/planners/fabrics/include/fab_core_util.h"
#include "mjpc/urdf_parser/include/joint.h"
#include "mjpc/urdf_parser/include/link.h"
#include "mjpc/urdf_parser/include/txml.h"

using namespace urdf;

template <typename T, typename TCollection>
static bool has_collection_element(const TCollection& collection, const T& elem) {
  return std::find(collection.begin(), collection.end(), elem) != collection.end();
}

int UrdfModel::get_dof() const {
  int dof = 0;
  for (const auto& join_name : active_joint_names) {
    if (has_collection_element(actuated_joint_names, join_name)) {
      dof++;
    }
  }
  return dof;
}

LinkPtr UrdfModel::get_link(const string& link_name) const {
  return link_map.contains(link_name) ? link_map.at(link_name) : nullptr;
}

JointPtr UrdfModel::get_joint(const string& joint_name) const {
  return joint_map.contains(joint_name) ? joint_map.at(joint_name) : nullptr;
}

std::vector<JointPtr> UrdfModel::get_joints(const std::string& base_name,
                                            const std::string& endtip_name) const {
  const auto chain = get_chain(base_name, endtip_name);
  std::vector<JointPtr> out_joint_list;

  for (const auto& item : chain) {
    if (joint_map.contains(item)) {
      auto joint = joint_map.at(item);
      if (has_collection_element(active_joint_names, joint->name)) {
        out_joint_list.push_back(joint);
      }
    }
  }
  return out_joint_list;
}

MaterialPtr UrdfModel::get_material(const string& mat_name) const {
  return material_map.contains(mat_name) ? material_map.at(mat_name) : nullptr;
}

std::vector<LinkPtr> UrdfModel::get_links() const { return fab_core::get_map_values<LinkPtr>(link_map); }

void UrdfModel::init_link_tree(map<string, string>& parent_link_tree) {
  for (const auto& [joint_name, joint] : joint_map) {
    string parent_link_name = joint->parent_link_name;
    string child_link_name = joint->child_link_name;

    if (parent_link_name.empty()) {
      ostringstream error_msg;
      error_msg << "Error while constructing model! Joint [" << joint_name
                << "] is missing a parent link specification.";
      throw URDFParseError(error_msg.str());
    }
    if (child_link_name.empty()) {
      ostringstream error_msg;
      error_msg << "Error while constructing model! Joint [" << joint_name
                << "] is missing a child link specification.";
      throw URDFParseError(error_msg.str());
    }

    auto child_link = get_link(child_link_name);
    if (child_link == nullptr) {
      ostringstream error_msg;
      error_msg << "Error while constructing model! Child link [" << child_link_name << "] of joint ["
                << joint_name << "] not found";
      throw URDFParseError(error_msg.str());
    }

    auto parent_link = get_link(parent_link_name);
    if (parent_link == nullptr) {
      ostringstream error_msg;
      error_msg << "Error while constructing model! Parent link [" << parent_link_name << "] of joint ["
                << joint_name << "] not found";
      throw URDFParseError(error_msg.str());
    }

    child_link->setParentLink(parent_link);
    child_link->setParentJoint(joint);

    parent_link->child_joints.push_back(joint);
    parent_link->child_links.push_back(child_link);

    parent_link_tree[child_link->name] = parent_link_name;
  }
}

void UrdfModel::findRoot(const map<string, string>& parent_link_tree) {
  for (const auto& [link_name, link] : link_map) {
    auto parent = parent_link_tree.find(link_name);
    if (parent == parent_link_tree.end()) {
      if (root_link == nullptr) {
        root_link = get_link(link_name);
      } else {
        ostringstream error_msg;
        error_msg << "Error! Multiple root links found: (" << root_link->name << ") and (" + link_name + ")!";
        throw URDFParseError(error_msg.str());
      }
    }
  }
  if (root_link == nullptr) {
    throw URDFParseError("Error! No root link found. The urdf does not contain a valid link tree.");
  }
}

bool UrdfModel::fromUrdfFile(const std::string& urdf_path) {
  std::ifstream stream(urdf_path.c_str());
  if (!stream) {
    std::cout << "URDF file " << urdf_path << " does not exist" << std::endl;
    return false;
  }
  return fromUrdfStr(std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()));
}

bool UrdfModel::fromUrdfStr(const std::string& xml_string) {
  TiXmlDocument xml_doc;
  xml_doc.Parse(xml_string.c_str());
  if (xml_doc.Error()) {
    std::string error_msg = xml_doc.ErrorDesc();
    xml_doc.ClearError();
    throw URDFParseError(error_msg);
  }

  // xml_doc.Print();
  TiXmlElement* robot_xml = xml_doc.RootElement();
  if (robot_xml == nullptr || robot_xml->ValueStr() != "robot") {
    std::string error_msg = "Error! Could not find the <robot> element in the xml file";
    throw URDFParseError(error_msg);
  }

  const char* model_name = robot_xml->Attribute("name");
  if (model_name != nullptr) {
    name = std::string(model_name);
  } else {
    std::string error_msg = "No name given for the robot. Please add a name attribute to the robot element!";
    throw URDFParseError(error_msg);
  }

  for (TiXmlElement* material_xml = robot_xml->FirstChildElement("material"); material_xml != nullptr;
       material_xml = material_xml->NextSiblingElement("material")) {
    auto material = Material::fromXml(material_xml, false);  // material needs to be fully defined here
    if (get_material(material->name) != nullptr) {
      std::ostringstream error_msg;
      error_msg << "Duplicate materials '" << material->name << "' found!";
      throw URDFParseError(error_msg.str());
    } else {
      material_map[material->name] = material;
    }
  }

  for (TiXmlElement* link_xml = robot_xml->FirstChildElement("link"); link_xml != nullptr;
       link_xml = link_xml->NextSiblingElement("link")) {
    auto link = Link::fromXml(link_xml);

    if (get_link(link->name) != nullptr) {
      std::ostringstream error_msg;
      error_msg << "Error! Duplicate links '" << link->name << "' found!";
      throw URDFParseError(error_msg.str());
    } else {
      // loop over link visual to find the materials
      if (!link->visuals.empty()) {
        for (const auto& visual : link->visuals) {
          if (!visual->material_name.empty()) {
            if (get_material(visual->material_name) != nullptr) {
              visual->material.emplace(get_material(visual->material_name));
            } else {
              // if no model material found use the one defined in the visual
              if (visual->material.has_value()) {
                material_map[visual->material_name] = visual->material.value();
              } else {
                // no material information available for this visual -> error
                std::ostringstream error_msg;
                error_msg << "Error! Link '" << link->name << "' material '" << visual->material_name
                          << " ' undefined!";
                throw URDFParseError(error_msg.str());
              }
            }
          }
        }
      }
      link_map[link->name] = link;
    }
  }

  if (link_map.empty()) {
    std::string error_msg = "Error! No link elements found in the urdf file.";
    throw URDFParseError(error_msg);
  }

  for (TiXmlElement* joint_xml = robot_xml->FirstChildElement("joint"); joint_xml != nullptr;
       joint_xml = joint_xml->NextSiblingElement("joint")) {
    auto joint = Joint::fromXml(joint_xml);

    if (get_joint(joint->name) != nullptr) {
      std::ostringstream error_msg;
      error_msg << "Error! Duplicate joints '" << joint->name << "' found!";
      throw URDFParseError(error_msg.str());
    } else {
      joint_map[joint->name] = joint;
      joint_list.push_back(joint);

      // [parent_name_map]
      const auto& child_link_name = joint->child_link_name;
      parent_name_map[child_link_name] = JointLinkNamePair{joint->name, joint->parent_link_name};

      // [child_name_map]
      const auto& parent_link_name = joint->parent_link_name;
      if (child_name_map.contains(parent_link_name)) {
        child_name_map[parent_link_name].emplace_back(joint->name, child_link_name);
      } else {
        child_name_map[parent_link_name] = {JointLinkNamePair{joint->name, child_link_name}};
      }
    }
  }

  std::map<std::string, std::string> parent_link_tree;
  init_link_tree(parent_link_tree);
  findRoot(parent_link_tree);
  {
    // 1-
    init_active_joints();
    // 2-
    init_actuated_joint_names();
    // 3-
    init_joint_name_map();
  }

#if URDF_MODEL_DEBUG_LOG
  print_self();
#endif
  return true;
}

void UrdfModel::print_self() const {
  fab_core::print_named_map2<JointLinkNamePair>(parent_name_map, "PARENT LINK NAME MAP");
  fab_core::print_named_map2<vector<JointLinkNamePair>>(child_name_map, "CHILD LINK NAME MAP");

  FAB_PRINT("ACTIVE JOINT NAMES: ", active_joint_names);
  fab_core::print_named_map2<int>(joint_name_map, "JOINT NAME MAP");
  FAB_PRINT("ACTUATED JOINT NAMES: ", actuated_joint_names);
}

std::vector<std::string> UrdfModel::get_chain(const std::string& base_name, const std::string& endtip_name,
                                              bool joints, bool links, bool fixed) const {
  std::vector<std::string> chain;
  if (links) {
    chain.push_back(endtip_name);
  }

  auto link_name = endtip_name;
  while (link_name != base_name) {
    auto& [joint_name, parent_link_name] = parent_name_map.at(link_name);

    if (joints && (fixed || joint_map.at(joint_name)->type != JointType::FIXED)) {
      chain.push_back(joint_name);
    }
    if (links) {
      chain.push_back(parent_link_name);
    }
    link_name = parent_link_name;
  }
  std::reverse(chain.begin(), chain.end());

#if URDF_MODEL_DEBUG_LOG
  std::cout << name << "'s CHAIN: " << base_name << "->" << endtip_name << ": " << fab_core::join(chain, ",")
            << std::endl;
#endif
  return chain;
}

void UrdfModel::init_joint_name_map() {
  int index = 0;
  for (const auto& joint_name : actuated_joint_names) {
    if (has_collection_element(active_joint_names, joint_name)) {
      joint_name_map.insert_or_assign(joint_name, index++);
    }
  }
}

void UrdfModel::init_actuated_joint_names() {
  for (const auto& joint : joint_list) {
    if (has_collection_element(actuated_joint_types, joint->type)) {
      actuated_joint_names.push_back(joint->name);
    }
  }
}

void UrdfModel::init_active_joints() {
  for (const auto& endtip : endtip_names) {
    auto parent_link_name = endtip;
    while (!has_collection_element(vector{base_link_name, root_link->name}, parent_link_name)) {
      const auto& [joint_name, link_name] = parent_name_map[parent_link_name];
      parent_link_name = link_name;
      active_joint_names.push_back(joint_name);
      if (parent_link_name == root_link->name) {
        break;
      }
    }
  }
}
