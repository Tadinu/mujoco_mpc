#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#ifndef M_PI
#define M_PI 3.141592538
#endif  // M_PI

#include "absl/strings/ascii.h"
#include "mjpc/urdf_parser/include/exception.h"
#include "mjpc/urdf_parser/include/txml.h"

using namespace std;

namespace urdf {
static inline std::string get_xml_attr(TiXmlElement* xml, const char* attr) {
  const char* attr_val = xml->Attribute(attr);
  return std::string(attr_val ? absl::StripAsciiWhitespace(std::string(attr_val)) : "");
}

struct Vector3 {
  double x;
  double y;
  double z;

  void clear() {
    x = 0.;
    y = 0.;
    z = 0.;
  }

  Vector3(double x, double y, double z) : x(x), y(y), z(z) {}

  Vector3(const Vector3& other) : x(other.x), y(other.y), z(other.z) {}

  Vector3() : x(0.), y(0.), z(0.) {}

  Vector3 operator+(const Vector3& other) const;
  Vector3 operator*(const double scale) const;
  friend Vector3 operator*(const double scale, const Vector3& v) { return v * scale; }
  double operator[](const int idx) const { return (idx == 0) ? x : (idx == 1) ? y : (idx == 2) ? z : -1; }
  bool operator==(const Vector3& other) const { return (x == other.x) && (y == other.y) && (z == other.z); }

  std::vector<double> to_vector() const {
    std::vector<double> res;
    res.push_back(x);
    res.push_back(y);
    res.push_back(z);
    return res;
  }

  std::string to_string() const {
    return std::string("[") + std::to_string(x) + std::string(",") + std::to_string(y) + std::string(",") +
           std::to_string(z) + std::string("]");
  }

  static Vector3 fromVecStr(const string& vector_str);
  static Vector3 Zero;
  static Vector3 UnitX;
  static Vector3 UnitY;
  static Vector3 UnitZ;
};

struct Rotation {
  double x;
  double y;
  double z;
  double w;
  Vector3 rpy;

  void clear() {
    x = 0.;
    y = 0.;
    z = 0.;
    w = 1.;
  }

  void set_rpy(double& roll, double& pitch, double& yaw);
  void normalize();
  Rotation get_inverse() const;

  Rotation operator*(const Rotation& other) const;
  Vector3 operator*(const Vector3& vec) const;

  double operator[](const int idx) const {
    return (idx == 0) ? rpy.x : (idx == 1) ? rpy.y : (idx == 2) ? rpy.z : -1;
  }

  Rotation() : x(0.), y(0.), z(0.), w(1.) { set_rpy(rpy.x, rpy.y, rpy.z); }
  Rotation(double x, double y, double z, double w) : x(x), y(y), z(z), w(w) { set_rpy(rpy.x, rpy.y, rpy.z); }

  Rotation(const Rotation& other) : x(other.x), y(other.y), z(other.z), w(other.w) {
    set_rpy(rpy.x, rpy.y, rpy.z);
  }

  std::string to_string() const { return rpy.to_string(); }

  static Rotation fromRpy(double roll, double pitch, double yaw);
  static Rotation fromRpyStr(const string& rotation_str);
  static Rotation Zero;
};

struct Color {
  float r;
  float g;
  float b;
  float a;

  void clear() {
    r = 0.;
    g = 0.;
    b = 0.;
    a = 1.;
  }

  Color() : r(0.), g(0.), b(0.), a(1.) {}

  Color(float r, float g, float b, float a) : r(r), g(g), b(b), a(a) {}

  Color(const Color& other) : r(other.r), g(other.g), b(other.b), a(other.a) {}

  static Color fromColorStr(const std::string& vector_str);
};

struct Transform {
  Vector3 position;
  Rotation rotation;

  void clear() {
    this->position.clear();
    this->rotation.clear();
  };

  Transform() : position(Vector3()), rotation(Rotation()) {}

  Transform(const Transform& other) : position(other.position), rotation(other.rotation) {}

  static Transform fromXml(TiXmlElement* xml);
};

struct Twist {
  Vector3 linear;
  Vector3 angular;

  void clear() {
    this->linear.clear();
    this->angular.clear();
  }

  Twist() : linear(Vector3()), angular(Vector3()) {}

  Twist(const Twist& other) : linear(other.linear), angular(other.angular) {}
};
}  // namespace urdf
