#include "mjpc/urdf_parser/include/link.h"

// abseil
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"

// urdf_parser
#include "absl/types/bad_any_cast.h"
#include "mjpc/urdf_parser/include/txml.h"

namespace urdf {
std::shared_ptr<Material> Material::fromXml(TiXmlElement* xml, bool only_name_is_ok) {
  bool has_rgb = false;
  bool has_filename = false;

  std::shared_ptr<Material> m = std::make_shared<Material>();

  auto name_str = xml->Attribute("name");
  if (name_str != nullptr) {
    m->name = name_str;
  } else {
    std::ostringstream error_msg;
    error_msg << "Error! Material without a name attribute detected!";
    throw URDFParseError(error_msg.str());
  }

  auto t = xml->FirstChildElement("texture");
  if (t != nullptr) {
    if (t->Attribute("filename") != nullptr) {
      m->texture_filename = t->Attribute("filename");
      has_filename = true;
    }
  }

  auto c = xml->FirstChildElement("color");
  if (c != nullptr) {
    if (c->Attribute("rgba") != nullptr) {
      try {
        m->color = Color::fromColorStr(c->Attribute("rgba"));
        has_rgb = true;
      } catch (URDFParseError& e) {
        std::ostringstream error_msg;
        error_msg << "Material [" << m->name << "] has malformed color rgba values: " << e.what() << "!";
        throw URDFParseError(error_msg.str());
      }
    }
  }

  if (!has_rgb && !has_filename) {
    if (!only_name_is_ok) // no need for an error if only name is ok
    {
      std::ostringstream error_msg;
      error_msg << "Material [" << m->name << "] has neither a texture nor a color defined!";
      throw URDFParseError(error_msg.str());
    }
  }

  return m;
}

const char* getParentLinkName(TiXmlElement* c) {
  TiXmlElement* e = c->Parent()->ToElement();
  while (e->Parent() != nullptr) {
    if (e->ValueStr() == "link") {
      break;
    }
    e = e->Parent()->ToElement();
  }
  return e->Attribute("name");
}

Inertial Inertial::fromXml(TiXmlElement* xml) {
  Inertial i;

  TiXmlElement* o = xml->FirstChildElement("origin");
  if (o != nullptr) {
    i.origin = Transform::fromXml(o);
  }

  TiXmlElement* mass_xml = xml->FirstChildElement("mass");
  if (mass_xml != nullptr) {
    const char* mass = mass_xml->Attribute("value");
    if (mass != nullptr) {
      if (!absl::SimpleAtod(std::string(mass), &i.mass)) {
        std::ostringstream error_msg;
        error_msg << "Error while parsing link '" << getParentLinkName(xml) << "': inertial mass ["
            << mass << "] is not a valid double!";
        throw URDFParseError(error_msg.str());
      }
    } else {
      std::ostringstream error_msg;
      error_msg << "Error while parsing link '" << getParentLinkName(xml)
          << "' <mass> element must have a value attribute!";
      throw URDFParseError(error_msg.str());
    }
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing link '" << getParentLinkName(xml)
        << "' inertial element must have a <mass> element!";
    throw URDFParseError(error_msg.str());
  }

  TiXmlElement* inertia_xml = xml->FirstChildElement("inertia");
  if (inertia_xml != nullptr) {
#define VERIFY_INERTIA(i_attr, out_i) \
    { \
      const auto i_val_str = std::string(inertia_xml->Attribute(i_attr)); \
      if (i_val_str.empty()) { \
        throw URDFParseError(std::string(getParentLinkName(xml)) + "<inertia>: " + i_attr + " is absent"); \
      } \
      if (!absl::SimpleAtod(i_val_str, &out_i)) { \
        throw URDFParseError(i_val_str + ": not a double");\
      } \
    }
    VERIFY_INERTIA("ixx", i.ixx);
    VERIFY_INERTIA("ixy", i.ixy);
    VERIFY_INERTIA("ixz", i.ixz);
    VERIFY_INERTIA("iyy", i.iyy);
    VERIFY_INERTIA("iyz", i.iyz);
    VERIFY_INERTIA("izz", i.izz);
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing link '" << getParentLinkName(xml)
        << "' inertial element must have a <inertia> element!";
    throw URDFParseError(error_msg.str());
  }

  return i;
}

std::shared_ptr<Visual> Visual::fromXml(TiXmlElement* xml) {
  std::shared_ptr<Visual> vis = std::make_shared<Visual>();

  TiXmlElement* o = xml->FirstChildElement("origin");
  if (o != nullptr) {
    try {
      vis->origin = Transform::fromXml(o);
    } catch (URDFParseError& e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing link '" << getParentLinkName(xml)
          << "': visual origin is not valid: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  TiXmlElement* geom = xml->FirstChildElement("geometry");
  if (geom != nullptr) {
    vis->geometry = Geometry::fromXml(geom);
  }

  const char* name_char = xml->Attribute("name");
  if (name_char != nullptr) {
    vis->name = name_char;
  }

  TiXmlElement* mat = xml->FirstChildElement("material");
  if (mat != nullptr) {
    if (mat->Attribute("name") != nullptr) {
      vis->material_name = mat->Attribute("name");
    } else {
      std::ostringstream error_msg;
      error_msg << "Error while parsing link '" << getParentLinkName(xml)
          << "': visual material must contain a name attribute!";
      throw URDFParseError(error_msg.str());
    }

    vis->material = Material::fromXml(mat, true);
  }

  return vis;
}

std::shared_ptr<Collision> Collision::fromXml(TiXmlElement* xml) {
  std::shared_ptr<Collision> col = std::make_shared<Collision>();

  TiXmlElement* o = xml->FirstChildElement("origin");
  if (o != nullptr) {
    try {
      col->origin = Transform::fromXml(o);
    } catch (URDFParseError& e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing link '" << getParentLinkName(xml)
          << "': collision origin is not a valid: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  }

  TiXmlElement* geom = xml->FirstChildElement("geometry");
  if (geom != nullptr) {
    col->geometry = Geometry::fromXml(geom);
  }

  const char* name_char = xml->Attribute("name");
  if (name_char != nullptr) {
    col->name = name_char;
  }

  return col;
}

std::shared_ptr<Link> Link::fromXml(TiXmlElement* xml) {
  std::shared_ptr<Link> link = std::make_shared<Link>();

  const char* name_char = xml->Attribute("name");
  if (name_char != nullptr) {
    link->name = std::string(name_char);
  } else {
    std::ostringstream error_msg;
    error_msg << "Error! Link without a name attribute detected!";
    throw URDFParseError(error_msg.str());
  }

  TiXmlElement* i = xml->FirstChildElement("inertial");
  if (i != nullptr) {
    link->inertial = Inertial::fromXml(i);
  }

  for (TiXmlElement* vis_xml = xml->FirstChildElement("visual"); vis_xml != nullptr;
       vis_xml = vis_xml->NextSiblingElement("visual")) {
    auto vis = Visual::fromXml(vis_xml);
    link->visuals.push_back(vis);
  }

  for (TiXmlElement* col_xml = xml->FirstChildElement("collision"); col_xml != nullptr;
       col_xml = col_xml->NextSiblingElement("collision")) {
    auto col = Collision::fromXml(col_xml);
    link->collisions.push_back(col);
  }

  return link;
}
} // namespace urdf
