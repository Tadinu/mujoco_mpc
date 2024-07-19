#include "mjpc/urdf_parser/include/geometry.h"

// abseil
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"

// urdf_parser
#include "mjpc/urdf_parser/include/link.h"
using namespace urdf;
using namespace std;

std::shared_ptr<Sphere> Sphere::fromXml(TiXmlElement* xml) {
  auto s = std::make_shared<Sphere>();

  const auto radius = urdf::get_xml_attr(xml, "radius");
  if (!radius.empty()) {
    if (!absl::SimpleAtod(radius, &s->radius)) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing link '" << getParentLinkName(xml) << "': sphere radius ["
          << xml->Attribute("radius") << "] is not a valid double!";
      throw URDFParseError(error_msg.str());
    }
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing link '" << getParentLinkName(xml)
        << "': Sphere shape must have a radius attribute";
    throw URDFParseError(error_msg.str());
  }

  return s;
}

std::shared_ptr<Box> Box::fromXml(TiXmlElement* xml) {
  auto b = std::make_shared<Box>();

  if (xml->Attribute("size") != nullptr) {
    try {
      string r = xml->Attribute("size");
      r = absl::StripAsciiWhitespace(r); // get rid of surrounding whitespace
      b->dim = Vector3::fromVecStr(r);
    } catch (URDFParseError& e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing link '" << getParentLinkName(xml) << "': box size ["
          << xml->Attribute("size") << "] is not a valid: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing link '" << getParentLinkName(xml)
        << "': Sphere shape must have a size attribute";
    throw URDFParseError(error_msg.str());
  }

  return b;
}

std::shared_ptr<Cylinder> Cylinder::fromXml(TiXmlElement* xml) {
  auto y = std::make_shared<Cylinder>();

  const auto length = urdf::get_xml_attr(xml, "length");
  if (!length.empty()) {
    if (!absl::SimpleAtod(length, &y->length)) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing link '" << getParentLinkName(xml) << "': cylinder length ["
          << xml->Attribute("length") << "] is not a valid double!";
      throw URDFParseError(error_msg.str());
    }
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing link '" << getParentLinkName(xml)
        << "': Cylinder shape length is absent!";
    throw URDFParseError(error_msg.str());
  }

  const auto radius = urdf::get_xml_attr(xml, "radius");
  if (!radius.empty()) {
    if (!absl::SimpleAtod(radius, &y->radius)) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing link '" << getParentLinkName(xml) << "': cylinder radius ["
          << xml->Attribute("radius") << "] is not a valid double!";
      throw URDFParseError(error_msg.str());
    }
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing link '" << getParentLinkName(xml)
        << "': Cylinder shape radius is absent!";
    throw URDFParseError(error_msg.str());
  }

  return y;
}

std::shared_ptr<Capsule> Capsule::fromXml(TiXmlElement* xml) {
  auto y = std::make_shared<Capsule>();

  const auto length = urdf::get_xml_attr(xml, "length");
  if (!length.empty()) {
    if (!absl::SimpleAtod(length, &y->length)) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing link '" << getParentLinkName(xml) << "': Capsule length ["
          << xml->Attribute("length") << "] is not a valid double!";
      throw URDFParseError(error_msg.str());
    }
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing link '" << getParentLinkName(xml)
        << "': Capsule shape length is absent!";
    throw URDFParseError(error_msg.str());
  }

  const auto radius = urdf::get_xml_attr(xml, "radius");
  if (!radius.empty()) {
    if (!absl::SimpleAtod(radius, &y->radius)) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing link '" << getParentLinkName(xml) << "': Capsule radius ["
          << xml->Attribute("radius") << "] is not a valid double!";
      throw URDFParseError(error_msg.str());
    }
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing link '" << getParentLinkName(xml)
        << "': Capsule shape radius is absent!";
    throw URDFParseError(error_msg.str());
  }

  return y;
}

std::shared_ptr<Mesh> Mesh::fromXml(TiXmlElement* xml) {
  auto m = std::make_shared<Mesh>();

  if (xml->Attribute("filename") != nullptr) {
    m->filename = xml->Attribute("filename");
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing link '" << getParentLinkName(xml)
        << "Mesh must contain a filename attribute!";
    throw URDFParseError(error_msg.str());
  }

  if (xml->Attribute("scale") != nullptr) {
    try {
      string r = xml->Attribute("scale");
      r = absl::StripAsciiWhitespace(r); // get rid of surrounding whitespace
      m->scale = Vector3::fromVecStr(r);
    } catch (URDFParseError& e) {
      std::ostringstream error_msg;
      error_msg << "Error while parsing link '" << getParentLinkName(xml) << "': mesh scale ["
          << xml->Attribute("scale") << "] is not a valid: " << e.what() << "!";
      throw URDFParseError(error_msg.str());
    }
  }
  return m;
}

std::shared_ptr<Geometry> Geometry::fromXml(TiXmlElement* xml) {
  if (xml == nullptr) {
    std::ostringstream error_msg;
    error_msg << "Error while parsing link '" << getParentLinkName(xml)
        << "' geometry structure pointer is null nothing to parse!";
    throw URDFParseError(error_msg.str());
  }

  TiXmlElement* shape = xml->FirstChildElement();
  if (shape == nullptr) {
    std::ostringstream error_msg;
    error_msg << "Error while parsing link '" << getParentLinkName(xml)
        << "' geometry does not contain any shape information!";
    throw URDFParseError(error_msg.str());
  }

  const std::string type_name = shape->ValueTStr().c_str();
  if (type_name == "sphere") {
    return Sphere::fromXml(shape);
  } else if (type_name == "box") {
    return Box::fromXml(shape);
  } else if (type_name == "cylinder") {
    return Cylinder::fromXml(shape);
  } else if (type_name == "capsule") {
    return Capsule::fromXml(shape);
  } else if (type_name == "mesh") {
    return Mesh::fromXml(shape);
  } else {
    std::ostringstream error_msg;
    error_msg << "Error while parsing link '" << getParentLinkName(xml) << "' unknown shape type '"
        << type_name << "'!";
    throw URDFParseError(error_msg.str());
  }
}
