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

LinkPtr UrdfModel::get_link(const string& name) {
  return link_map.contains(name) ? link_map.at(name) : nullptr;
}

JointPtr UrdfModel::get_joint(const string& name) {
  return joint_map.contains(name) ? joint_map.at(name) : nullptr;
}

std::vector<JointPtr> UrdfModel::get_joints(const std::string& base_name, const std::string& endtip_name) {
  const auto chain = get_chain(base_name, endtip_name);
  std::vector<JointPtr> joint_list;

  for (const auto& item : chain) {
    if (joint_map.contains(item)) {
      auto joint = joint_map[item];
      if (has_collection_element(active_joint_names, joint->name)) {
        joint_list.push_back(joint);
      }
    }
  }
  return joint_list;
}

MaterialPtr UrdfModel::get_material(const string& name) {
  if (material_map.find(name) == material_map.end()) {
    return nullptr;
  } else {
    return material_map.find(name)->second;
  }
}

std::vector<std::shared_ptr<Link>> UrdfModel::get_links() const {
  std::vector<std::shared_ptr<Link>> links;
  for (auto link = link_map.begin(); link != link_map.end(); link++) {
    links.push_back(link->second);
  }
  return links;
}

void UrdfModel::init_link_tree(map<string, string>& parent_link_tree) {
  for (auto joint = joint_map.begin(); joint != joint_map.end(); joint++) {
    string parent_link_name = joint->second->parent_link_name;
    string child_link_name = joint->second->child_link_name;

    if (parent_link_name.empty()) {
      ostringstream error_msg;
      error_msg << "Error while constructing model! Joint [" << joint->first
                << "] is missing a parent link specification.";
      throw URDFParseError(error_msg.str());
    }
    if (child_link_name.empty()) {
      ostringstream error_msg;
      error_msg << "Error while constructing model! Joint [" << joint->first
                << "] is missing a child link specification.";
      throw URDFParseError(error_msg.str());
    }

    auto child_link = get_link(child_link_name);
    if (child_link == nullptr) {
      ostringstream error_msg;
      error_msg << "Error while constructing model! Child link [" << child_link_name << "] of joint ["
                << joint->first << "] not found";
      throw URDFParseError(error_msg.str());
    }

    auto parent_link = get_link(parent_link_name);
    if (parent_link == nullptr) {
      ostringstream error_msg;
      error_msg << "Error while constructing model! Parent link [" << parent_link_name << "] of joint ["
                << joint->first << "] not found";
      throw URDFParseError(error_msg.str());
    }

    child_link->setParentLink(parent_link);
    child_link->setParentJoint(joint->second);

    parent_link->child_joints.push_back(joint->second);
    parent_link->child_links.push_back(child_link);

    parent_link_tree[child_link->name] = parent_link_name;
  }
}

void UrdfModel::findRoot(const map<string, string>& parent_link_tree) {
  for (auto l = link_map.begin(); l != link_map.end(); l++) {
    auto parent = parent_link_tree.find(l->first);
    if (parent == parent_link_tree.end()) {
      if (root_link == nullptr) {
        root_link = get_link(l->first);
      } else {
        ostringstream error_msg;
        error_msg << "Error! Multiple root links found: (" << root_link->name << ") and (" + l->first + ")!";
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
        for (auto visual : link->visuals) {
          if (!visual->material_name.empty()) {
            if (get_material(visual->material_name) != nullptr) {
              visual->material.emplace(get_material(visual->material_name.c_str()));
            } else {
              // if no model matrial found use the one defined in the visual
              if (visual->material.has_value()) {
                material_map[visual->material_name] = visual->material.value();
              } else {
                // no matrial information available for this visual -> error
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

  if (link_map.size() == 0) {
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
  std::cout << "PARENT NAME MAP" << std::endl;
  for (const auto& [name, pair] : parent_name_map) {
    std::cout << name << ":" << pair.first << "-" << pair.second << std::endl;
  }

  std::cout << "CHILD NAME MAP" << std::endl;
  for (const auto& [name, pairs] : child_name_map) {
    for (const auto& pair : pairs) {
      std::cout << name << ":" << pair.first << "-" << pair.second << std::endl;
    }
  }

  std::cout << "ACTIVE JOINT NAMES: " << fab_core::join(active_joint_names, ",") << std::endl;
  std::cout << "JOINT NAME MAP" << std::endl;
  for (const auto& [name, idx] : joint_name_map) {
    std::cout << name << ":" << idx << std::endl;
  }

  std::cout << "ACTUATED JOINT NAMES: " << fab_core::join(actuated_joint_names, ",") << std::endl;
}

std::vector<std::string> UrdfModel::get_chain(const std::string& base_name, const std::string& endtip_name,
                                              bool joints, bool links, bool fixed) {
  std::vector<std::string> chain;
  if (links) {
    chain.push_back(endtip_name);
  }

  auto link_name = endtip_name;
  while (link_name != base_name) {
    auto& [joint_name, parent_link_name] = parent_name_map[link_name];

    if (joints && (fixed || joint_map[joint_name]->type != JointType::FIXED)) {
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
  for (const auto& [joint_name, joint] : joint_map) {
    if (has_collection_element(actuated_joint_types, joint->type)) {
      actuated_joint_names.push_back(joint_name);
    }
  }
}

void UrdfModel::init_active_joints() {
  for (const auto& endtip : endtip_names) {
    auto parent_link_name = endtip;
    while (!has_collection_element(vector<string>{base_link_name, root_link->name}, parent_link_name)) {
      auto& [joint_name, link_name] = parent_name_map[parent_link_name];
      parent_link_name = link_name;
      active_joint_names.push_back(joint_name);
      if (parent_link_name == root_link->name) {
        break;
      }
    }
  }
}
