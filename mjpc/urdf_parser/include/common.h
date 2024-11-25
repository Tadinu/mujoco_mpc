#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>

// MuJoCo
#include <mujoco/mujoco.h>

// Mjpc
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
  double x = 0.;
  double y = 0.;
  double z = 0.;

  void clear() {
    x = 0.;
    y = 0.;
    z = 0.;
  }

  Vector3() = default;
  explicit Vector3(const std::vector<double>& xyz) : x(xyz[0]), y(xyz[1]), z(xyz[2]) {}
  Vector3(double x, double y, double z) : x(x), y(y), z(z) {}
  Vector3(const Vector3& other) = default;
  explicit Vector3(const mjtNum mj_pos[3]) : x(mj_pos[0]), y(mj_pos[1]), z(mj_pos[2]) {}

  Vector3 operator+(const Vector3& other) const;
  void operator+=(const Vector3& other) { *this = *this + other; }
  Vector3 operator*(double scale) const;
  friend Vector3 operator*(const double scale, const Vector3& v) { return v * scale; }
  double operator[](const int idx) const { return (idx == 0) ? x : (idx == 1) ? y : (idx == 2) ? z : -1; }
  bool operator==(const Vector3& other) const { return (x == other.x) && (y == other.y) && (z == other.z); }

  std::vector<double> to_vector() const { return {x, y, z}; }

  std::string to_string() const {
    return "[" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + "]";
  }

  static Vector3 fromVecStr(const string& vector_str);
  static Vector3 Zero;
  static Vector3 UnitX;
  static Vector3 UnitY;
  static Vector3 UnitZ;
};

struct Rotation {
  double x = 0.;
  double y = 0.;
  double z = 0.;
  double w = 1.;
  Vector3 rpy;

  void clear() {
    x = 0.;
    y = 0.;
    z = 0.;
    w = 1.;
  }

  void set_rpy();
  void normalize();
  Rotation get_inverse() const;

  Rotation operator*(const Rotation& other) const;
  Vector3 operator*(const Vector3& vec) const;

  double operator[](const int idx) const { return rpy[idx]; }

  Rotation() = default;
  Rotation(const double x, const double y, const double z, const double w) : x(x), y(y), z(z), w(w) {
    set_rpy();
  }
  Rotation(const Rotation& other) : x(other.x), y(other.y), z(other.z), w(other.w) { set_rpy(); }
  explicit Rotation(const mjtNum mj_pos[4]) : x(mj_pos[1]), y(mj_pos[2]), z(mj_pos[3]), w(mj_pos[0]) {
    set_rpy();
  }

  std::vector<double> to_quat() const { return {w, x, y, z}; }
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

  Color() = default;
  Color(float r, float g, float b, float a) : r(r), g(g), b(b), a(a) {}

  explicit Color(const Color& other) : r(other.r), g(other.g), b(other.b), a(other.a) {}

  static Color fromColorStr(const std::string& vector_str);
};

struct Transform {
  Vector3 position;
  Rotation rotation;

  void clear() {
    this->position.clear();
    this->rotation.clear();
  };

  Transform operator*(const Transform& other) const {
    mjtNum pos[3];
    mjtNum quat[4];
    mju_mulPose(pos, quat, position.to_vector().data(), rotation.to_quat().data(),
                other.position.to_vector().data(), other.rotation.to_quat().data());
    return Transform{.position = Vector3(pos), .rotation = Rotation(quat)};
  }
  static Transform fromXml(TiXmlElement* xml);
};

struct Twist {
  Vector3 linear;
  Vector3 angular;

  void clear() {
    this->linear.clear();
    this->angular.clear();
  }

  Twist() = default;
  Twist(const Twist& other) : linear(other.linear), angular(other.angular) {}
};
}  // namespace urdf
