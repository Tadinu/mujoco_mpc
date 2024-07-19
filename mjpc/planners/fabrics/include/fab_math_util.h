#pragma once

#include <stdexcept>
#include <vector>

// casadi
#include <casadi/casadi.hpp>
#include <casadi/core/casadi_types.hpp>

// fabrics
#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_core_util.h"
#include "mjpc/urdf_parser/include/common.h"

namespace fab_math {
static CaSX CASX_TRANSF_IDENTITY = CaSX::eye(4);
static CaSX outer_product(const CaSX& a, const CaSX& b) {
  const auto m = a.size().first;
  const auto A = CaSX(CaSX::repmat(a.T(), m)).T();
  const auto B = CaSX::repmat(b.T(), m);
  return CaSX::times(A, B);
}

static CaSX closest_point_to_line(const CaSX& point, const CaSX& line_start, const CaSX& line_end) {
  const auto line_vector = line_end - line_start;
  const auto point_vector = point - line_start;
  auto t = CaSX::dot(point_vector, line_vector) / CaSX::dot(line_vector, line_vector);
  t = CaSX::fmax(0, CaSX::fmin(1, t));
  return line_start + t * line_vector;
}

static CaSX clamp(const CaSX& a, const float a_min, const float a_max) {
  return CaSX::fmin(a_max, CaSX::fmax(a, a_min));
}

static CaSX point_to_point(const CaSX& point_1, const CaSX& point_2) {
  return CaSX::norm_2(point_1 - point_2);
}

static CaSX sphere_to_point(const CaSX& sphere_center, const CaSX& point, const CaSX& sphere_radius) {
  return point_to_point(sphere_center, point) - sphere_radius;
}

static CaSX sphere_to_sphere(const CaSX& sphere_1_center, const CaSX& sphere_2_center,
                             const CaSX& sphere_1_radius, const CaSX& sphere_2_radius) {
  const CaSX distance = point_to_point(sphere_1_center, sphere_2_center);
  return distance - (sphere_1_radius - sphere_2_radius);
}

static CaSX point_to_line(const CaSX& point, const CaSX& line_start, const CaSX& line_end) {
  const CaSX line_vec = line_end - line_start;
  const CaSX point_vec = point - line_start;
  const CaSX proj_length = CaSX::dot(point_vec, line_vec) / CaSX::norm_2(line_vec);
  const CaSX distance_0 = CaSX::norm_2(point - line_start);
  const CaSX distance_1 = CaSX::norm_2(point - line_end);
  const CaSX proj_point = line_start + proj_length * line_vec / CaSX::norm_2(line_vec);
  const CaSX distance_2 = CaSX::norm_2(point - proj_point);

  return CaSX::if_else(proj_length <= 0, distance_0,
                       CaSX::if_else(proj_length >= CaSX::norm_2(line_vec), distance_1, distance_2));
}

/*
 * Computes the distance between two lines according to
 * Real-Time Collision Detection by Christer Ericson, p148
 */
static CaSX line_to_line(const CaSX& line_1_start, const CaSX& line_1_end, const CaSX& line_2_start,
                         const CaSX& line_2_end) {
  static constexpr double eps = 1e-5;
  const CaSX d1 = line_1_end - line_1_start;
  const CaSX d2 = line_2_end - line_2_start;
  const CaSX r = line_1_start - line_2_start;
  const CaSX a = CaSX::dot(d1, d1);
  const CaSX e = CaSX::dot(d2, d2);
  const CaSX f = CaSX::dot(d2, r);
  const CaSX c = CaSX::dot(d1, r);
  const CaSX b = CaSX::dot(d1, d2);
  const CaSX denom = a * e - b * b;
  const CaSX s = CaSX::if_else(
      a <= eps, 0.0,
      CaSX::if_else(e <= eps, clamp(-c / a, 0.0, 1.0),
                    CaSX::if_else(denom != 0.0, clamp((b * f - c * e) / denom, 0.0, 1.0), 0.0)));

  const CaSX t =
      CaSX::if_else(a <= eps, clamp(f / e, 0.0, 1.0), CaSX::if_else(e <= eps, 0.0, (b * s + f) / e));
  const CaSX s_1 = CaSX::if_else(t < 0.0, clamp(-c / a, 0.0, 1.0), s);
  const CaSX s_2 = CaSX::if_else(t > f, clamp((b * f - c * e) / denom, 0.0, 1.0), s_1);
  const CaSX t_1 = clamp(t, 0.0, 1.0);
  const CaSX c1 = line_1_start + d1 * s_2;
  const CaSX c2 = line_2_start + d2 * t_1;
  const CaSX distance = CaSX::if_else(CaSX::logic_and(a <= eps, e <= eps),
                                      CaSX::dot(line_1_start - line_2_start, line_1_start - line_2_start),
                                      CaSX::dot(c1 - c2, c1 - c2));
  // CaSX::vertcat(CaSX::sqrt(distance), t_1, s_2, c1, c2)
  return CaSX::sqrt(distance);
}

static CaSX point_to_plane(const CaSX& point, const CaSX& plane) {
  CaSX plane_0_3;
  plane.get(plane_0_3, false, CaSlice(0, 3));
  CaSX plane_3;
  plane.get(plane_3, false, CaSlice(3));
  const CaSX distance = CaSX::abs(CaSX::dot(plane_0_3, point) + plane_3) / CaSX::norm_2(plane_0_3);
  return distance;
}

static CaSX sphere_to_plane(const CaSX& sphere_center, const CaSX& plane, const CaSX& sphere_radius) {
  return point_to_plane(sphere_center, plane) - sphere_radius;
}

// Assume no intersection between the line and the plane
static CaSX line_to_plane(const CaSX& line_start, const CaSX& line_end, const CaSX& plane) {
  CaSX plane_0_3;
  plane.get(plane_0_3, false, CaSlice(0, 3));
  const CaSX distance_line_start = point_to_plane(line_start, plane);
  const CaSX distance_line_end = point_to_plane(line_end, plane);
  const CaSX min_distance_ends = CaSX::fmin(distance_line_start, distance_line_end);
  const CaSX product_dot_products = CaSX::dot(plane_0_3, line_start) * CaSX::dot(plane_0_3, line_end);
  return CaSX::if_else(product_dot_products < 0, 0.0, min_distance_ends);
}

static CaSX capsule_to_plane(const CaSXVector& capsule_center, const CaSX& plane,
                             const CaSX& capsule_radius) {
  return line_to_plane(capsule_center[0], capsule_center[1], plane) - capsule_radius;
}

static CaSX capsule_to_capsule(const CaSXVector& capsule_1_centers, const CaSXVector& capsule_2_centers,
                               const CaSX& capsule_1_radius, const CaSX& capsule_2_radius) {
  return line_to_line(capsule_1_centers[0], capsule_1_centers[1], capsule_2_centers[0],
                      capsule_2_centers[1]) -
         capsule_1_radius - capsule_2_radius;
}

static CaSX capsule_to_sphere(const CaSXVector& capsule_center, const CaSX& sphere_center,
                              const CaSX& capsule_radius, const CaSX& sphere_radius) {
  assert(capsule_center.size() == 2);
  const auto distance_line_center = point_to_line(sphere_center, capsule_center[0], capsule_center[1]);
  return CaSX::fmax(distance_line_center - capsule_radius - sphere_radius, 0.0);
}

static CaSXVector cuboid_to_point_half_distances(const CaSX& cuboid_center, const CaSX& cuboid_size,
                                                 const CaSX& point) {
  CaSXVector half_distances;
  for (auto i = 0; i < point.size().first; ++i) {
    CaSX point_i, cuboid_center_i, cuboid_size_i;
    point.get(point_i, false, CaSlice(i));
    cuboid_center.get(cuboid_center_i, false, CaSlice(i));
    cuboid_size.get(cuboid_size_i, false, CaSlice(i));
    half_distances.push_back(CaSX::fmax(CaSX::abs(point_i - cuboid_center_i) - 0.5 * cuboid_size_i, 0.0));
  }
  return half_distances;
}

static CaSX rectangle_to_point(const CaSX& rectangle_center, const CaSX& rectangle_size, const CaSX& point) {
  const auto half_distances = cuboid_to_point_half_distances(rectangle_center, rectangle_size, point);
  return CaSX::sqrt(CaSX::pow(half_distances[0], 2) + CaSX::pow(half_distances[1], 2));
}

static CaSX rectangle_to_line(const CaSX& rectangle_center, const CaSX& rectangle_size,
                              const CaSX& line_start, const CaSX& line_end) {
  CaSX min_distance = CaSX::fmin(rectangle_to_point(rectangle_center, rectangle_size, line_start),
                                 rectangle_to_point(rectangle_center, rectangle_size, line_end));
  for (const auto i : {-1, 1}) {
    for (const auto j : {-1, 1}) {
      const CaSX index = {i, j};
      const CaSX corner_transform = 0.5 * rectangle_size * index;
      const CaSX corner = rectangle_center + corner_transform;
      min_distance = CaSX::fmin(min_distance, point_to_line(corner, line_start, line_end));
    }
  }
  return min_distance;
}

static CaSX cuboid_to_point(const CaSX& cuboid_center, const CaSX& cuboid_size, const CaSX& point) {
  const auto half_distances = cuboid_to_point_half_distances(cuboid_center, cuboid_size, point);
  return CaSX::sqrt(CaSX::pow(half_distances[0], 2) + CaSX::pow(half_distances[1], 2) +
                    CaSX::pow(half_distances[2], 3));
}

static CaSX edge_of_cuboid(const CaSX& cuboid_center, const CaSX& cuboid_size, const int index) {
  const casadi::SXVectorVector edges = {
      {{-1, -1, -1}, {1, -1, -1}}, {{-1, -1, -1}, {-1, 1, -1}}, {{-1, -1, -1}, {-1, -1, 1}},
      {{-1, -1, 1}, {1, -1, 1}},   {{-1, -1, 1}, {-1, 1, 1}},   {{-1, 1, -1}, {1, 1, -1}},
      {{-1, 1, -1}, {-1, 1, 1}},   {{1, -1, -1}, {1, 1, -1}},   {{1, -1, -1}, {1, -1, 1}},
      {{1, 1, 1}, {-1, 1, 1}},     {{1, 1, 1}, {1, -1, 1}},     {{1, 1, 1}, {1, 1, -1}}};
  const CaSX edge_start = cuboid_center + 0.5 * cuboid_size * edges[index][0];
  const CaSX edge_end = cuboid_center + 0.5 * cuboid_size * edges[index][1];
  return CaSX::vertcat({edge_start, edge_end});
}

static CaSX cuboid_to_line(const CaSX& cuboid_center, const CaSX& cuboid_size, const CaSX& line_start,
                           const CaSX& line_end) {
  CaSX distance = CaSX::fmin(cuboid_to_point(cuboid_center, cuboid_size, line_start),
                             cuboid_to_point(cuboid_center, cuboid_size, line_end));
  for (auto i = 0; i < 12; ++i) {
    const CaSX edge = edge_of_cuboid(cuboid_center, cuboid_size, i);
    CaSX edge_0_3, edge_3_6;
    edge.get(edge_0_3, false, CaSlice(0, 3));
    edge.get(edge_3_6, false, CaSlice(3, 6));
    const CaSX edge_start = edge_0_3;
    const CaSX edge_end = edge_3_6;
    const CaSX distance_line_edge = line_to_line(edge_start, edge_end, line_start, line_end);
    distance = CaSX::fmin(distance, distance_line_edge);
  }
  return distance;
}

static CaSX cuboid_to_sphere(const CaSX& cuboid_center, const CaSX& sphere_center, const CaSX& cuboid_size,
                             const CaSX& sphere_size) {
  return CaSX::fmax(0.0, cuboid_to_point(cuboid_center, cuboid_size, sphere_center) - sphere_size);
}

static CaSX cuboid_to_capsule(const CaSX& cuboid_center, const CaSXVector& capsule_center,
                              const CaSX& cuboid_size, const CaSX& capsule_radius) {
  return CaSX::fmax(
      cuboid_to_line(cuboid_center, cuboid_size, capsule_center[0], capsule_center[1]) - capsule_radius, 0.0);
}

// ==========================================================================================================
// URDF-2-CASADI UTILS
//
#include "mjpc/urdf_parser/include/common.h"

static CaSX prismatic(const urdf::Vector3& xyz, const urdf::Vector3& rpy, const urdf::Vector3& axis,
                      const CaSX& qi) {
  CaSX T = CaSX::zeros(4, 4);

  // Origin rotation from RPY ZYX convention
  const auto cr = CaSX::cos(rpy[0]);
  const auto sr = CaSX::sin(rpy[0]);
  const auto cp = CaSX::cos(rpy[1]);
  const auto sp = CaSX::sin(rpy[1]);
  const auto cy = CaSX::cos(rpy[2]);
  const auto sy = CaSX::sin(rpy[2]);
  const auto r00 = cy * cp;
  const auto r01 = cy * sp * sr - sy * cr;
  const auto r02 = cy * sp * cr + sy * sr;
  const auto r10 = sy * cp;
  const auto r11 = sy * sp * sr + cy * cr;
  const auto r12 = sy * sp * cr - cy * sr;
  const auto r20 = -sp;
  const auto r21 = cp * sr;
  const auto r22 = cp * cr;
  const auto p0 = r00 * axis[0] * qi + r01 * axis[1] * qi + r02 * axis[2] * qi;
  const auto p1 = r10 * axis[0] * qi + r11 * axis[1] * qi + r12 * axis[2] * qi;
  const auto p2 = r20 * axis[0] * qi + r21 * axis[1] * qi + r22 * axis[2] * qi;

  // Homogeneous transformation matrix
  fab_core::set_casx2(T, 0, 0, r00);
  fab_core::set_casx2(T, 0, 0, r00);
  fab_core::set_casx2(T, 0, 1, r01);
  fab_core::set_casx2(T, 0, 2, r02);
  fab_core::set_casx2(T, 1, 0, r10);
  fab_core::set_casx2(T, 1, 1, r11);
  fab_core::set_casx2(T, 1, 2, r12);
  fab_core::set_casx2(T, 2, 0, r20);
  fab_core::set_casx2(T, 2, 1, r21);
  fab_core::set_casx2(T, 2, 2, r22);
  fab_core::set_casx2(T, 0, 3, xyz[0] + p0);
  fab_core::set_casx2(T, 1, 3, xyz[1] + p1);
  fab_core::set_casx2(T, 2, 3, xyz[2] + p2);
  fab_core::set_casx2(T, 3, 3, 1.0);
  return T;
}

static CaSX revolute(const urdf::Vector3& xyz, const urdf::Vector3& rpy, const urdf::Vector3& axis,
                     const CaSX& qi) {
  CaSX T = CaSX::zeros(4, 4);

  // Origin rotation from RPY ZYX convention
  const auto cr = CaSX::cos(rpy[0]);
  const auto sr = CaSX::sin(rpy[0]);
  const auto cp = CaSX::cos(rpy[1]);
  const auto sp = CaSX::sin(rpy[1]);
  const auto cy = CaSX::cos(rpy[2]);
  const auto sy = CaSX::sin(rpy[2]);
  const auto r00 = cy * cp;
  const auto r01 = cy * sp * sr - sy * cr;
  const auto r02 = cy * sp * cr + sy * sr;
  const auto r10 = sy * cp;
  const auto r11 = sy * sp * sr + cy * cr;
  const auto r12 = sy * sp * cr - cy * sr;
  const auto r20 = -sp;
  const auto r21 = cp * sr;
  const auto r22 = cp * cr;

  // joint rotation from skew sym axis angle
  const auto cqi = CaSX::cos(qi);
  const auto sqi = CaSX::sin(qi);
  const auto s00 = (1 - cqi) * axis[0] * axis[0] + cqi;
  const auto s11 = (1 - cqi) * axis[1] * axis[1] + cqi;
  const auto s22 = (1 - cqi) * axis[2] * axis[2] + cqi;
  const auto s01 = (1 - cqi) * axis[0] * axis[1] - axis[2] * sqi;
  const auto s10 = (1 - cqi) * axis[0] * axis[1] + axis[2] * sqi;
  const auto s12 = (1 - cqi) * axis[1] * axis[2] - axis[0] * sqi;
  const auto s21 = (1 - cqi) * axis[1] * axis[2] + axis[0] * sqi;
  const auto s20 = (1 - cqi) * axis[0] * axis[2] - axis[1] * sqi;
  const auto s02 = (1 - cqi) * axis[0] * axis[2] + axis[1] * sqi;

  // Homogeneous transformation matrix
  fab_core::set_casx2(T, 0, 0, r00 * s00 + r01 * s10 + r02 * s20);
  fab_core::set_casx2(T, 1, 0, r10 * s00 + r11 * s10 + r12 * s20);
  fab_core::set_casx2(T, 2, 0, r20 * s00 + r21 * s10 + r22 * s20);
  fab_core::set_casx2(T, 0, 1, r00 * s01 + r01 * s11 + r02 * s21);
  fab_core::set_casx2(T, 1, 1, r10 * s01 + r11 * s11 + r12 * s21);
  fab_core::set_casx2(T, 2, 1, r20 * s01 + r21 * s11 + r22 * s21);
  fab_core::set_casx2(T, 0, 2, r00 * s02 + r01 * s12 + r02 * s22);
  fab_core::set_casx2(T, 1, 2, r10 * s02 + r11 * s12 + r12 * s22);
  fab_core::set_casx2(T, 2, 2, r20 * s02 + r21 * s12 + r22 * s22);

  fab_core::set_casx2(T, 0, 3, xyz[0]);
  fab_core::set_casx2(T, 1, 3, xyz[1]);
  fab_core::set_casx2(T, 2, 3, xyz[2]);
  fab_core::set_casx2(T, 3, 3, 1.0);
  return T;
}

// Returns a rotation matrix from roll pitch yaw. ZYX convention
static CaSX rotation_rpy(const urdf::Vector3& rpy) {
  const auto r = rpy[0];
  const auto p = rpy[1];
  const auto y = rpy[2];

  const auto cr = CaSX::cos(r);
  const auto sr = CaSX::sin(r);
  const auto cp = CaSX::cos(p);
  const auto sp = CaSX::sin(p);
  const auto cy = CaSX::cos(y);
  const auto sy = CaSX::sin(y);

  return CaSX::blockcat({{cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr},
                         {sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr},
                         {-sp, cp * sr, cp * cr}});
}

// Homogeneous transformation matrix with roll pitch yaw
static CaSX transform(const urdf::Vector3& xyz, const urdf::Vector3& rpy) {
  CaSX T = CaSX::zeros(4, 4);
  fab_core::set_casx2(T, {std::numeric_limits<casadi_int>::min(), 3},
                      {std::numeric_limits<casadi_int>::min(), 3}, rotation_rpy(rpy));
  fab_core::set_casx2(T, {std::numeric_limits<casadi_int>::min(), 3}, 3, CaSX(xyz.to_vector()));
  fab_core::set_casx2(T, 3, 3, 1.0);
  return T;
}
}  // namespace fab_math
