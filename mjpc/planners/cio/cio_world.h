#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <random>
#include <utility>

// Eigen
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>

// MJPC
#include "mjpc/planners/cio/cio_common.h"
#include "mjpc/planners/cio/cio_util.h"
#include "mjpc/utilities.h"

class CIOObject {
public:
  CIOObject() = default;
  explicit CIOObject(int id, CIOPose pose = CIOPose(), CIOVelocity vel = CIOVelocity(),
                     double step_size = 0.5)
      : id_(id), pose_(std::move(pose)), vel_(std::move(vel)), step_size_(step_size) {}

  int id() const { return id_; }
  CIOPose pose() const { return pose_; }
  CIOVelocity vel() const { return vel_; }
  CIOAcceleration acc() const { return acc_; }
  void set_dynamics(const CIOPose& pose, const CIOVelocity& vel, const CIOAcceleration& acc) {
    pose_ = pose;
    vel_ = vel;
    acc_ = acc;
  }

  virtual bool check_collisions(const CIOObject& other_object) {
    // CIOTODO
    return false;
  }

  Eigen::Vector3d get_surface_normal(const Eigen::Vector3d& point) const {
    return (point - pose_.position()).normalized();
  }

  virtual Eigen::Vector3d project_point(const Eigen::Vector3d& point) const = 0;
  virtual void discretize() {}
  virtual bool check_inside(const Eigen::Vector3d& point) { return false; }

protected:
  int id_ = 0;
  CIOPose pose_;
  CIOVelocity vel_;
  CIOAcceleration acc_;
  double step_size_ = 0.001;
  double rad_bounds_ = 1e-1;
};
using CIOObjectPtr = std::shared_ptr<CIOObject>;

#if 1
// Ref: https://github.com/ctu-mrs/mrs_lib/blob/master/src/geometry/shapes.cpp
class CIOLine {
public:
  CIOLine() = default;
  CIOLine(Eigen::Vector3d p1, Eigen::Vector3d p2) : p1_(std::move(p1)), p2_(std::move(p2)) {}

  Eigen::Vector3d p1() const { return p1_; }
  Eigen::Vector3d p2() const { return p2_; }

  Eigen::Vector3d direction() const { return (p2_ - p1_); }

  CIOLine directionCast(Eigen::Vector3d origin, Eigen::Vector3d direction) {
    return CIOLine(std::move(origin), origin + std::move(direction));
  }

  Eigen::Vector3d project_point(const Eigen::Vector3d& p) const {
    const auto p1p = p - p1_;
    const auto p12 = p2_ - p1_;
    return p1_ + p1p.dot(p12) / p12.dot(p12) * p12;
  }

private:
  Eigen::Vector3d p1_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d p2_ = Eigen::Vector3d::Zero();
};

class CIOTriangle {
public:
  CIOTriangle(Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3d p3)
      : p1_(std::move(p1)), p2_(std::move(p2)), p3_(std::move(p3)) {}

  Eigen::Vector3d p1() const { return p1_; }
  Eigen::Vector3d p2() const { return p2_; }
  Eigen::Vector3d p3() const { return p3_; }

  Eigen::Vector3d normal() const {
    Eigen::Vector3d n;
    n = (p2_ - p1_).cross(p3_ - p1_);
    return n.normalized();
  }

  Eigen::Vector3d center() const { return (p1_ + p2_ + p3_) / 3.0; }

  std::vector<Eigen::Vector3d> vertices() const {
    std::vector<Eigen::Vector3d> vertices;
    vertices.push_back(p1_);
    vertices.push_back(p2_);
    vertices.push_back(p3_);
    return vertices;
  }

  Eigen::Vector3d intersection_ray(const CIOLine& r, double epsilon) const {
    // The Möller–Trumbore algorithm
    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    Eigen::Vector3d v1 = p2_ - p1_;
    Eigen::Vector3d v2 = p3_ - p1_;
    Eigen::Vector3d h = r.direction().cross(v2);
    double res = v1.dot(h);
    if (res > -epsilon && res < epsilon) {
      return Eigen::Vector3d::Zero();
    }
    double f = 1.0 / res;
    Eigen::Vector3d s = r.p1() - p1_;
    double u = f * s.dot(h);
    if (u < 0.0 || u > 1.0) {
      return Eigen::Vector3d::Zero();
    }
    Eigen::Vector3d q = s.cross(v1);
    double v = f * r.direction().dot(q);
    if (v < 0.0 || u + v > 1.0) {
      return Eigen::Vector3d::Zero();
    }
    double t = f * v2.dot(q);
    if (t > epsilon) {
      Eigen::Vector3d ret = r.p1() + r.direction() * t;
      return ret;
    }
    return Eigen::Vector3d::Zero();
  }

  Eigen::Vector3d p1_ = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector3d p2_ = Eigen::Vector3d(1, 0, 0);
  Eigen::Vector3d p3_ = Eigen::Vector3d(0, 0, 1);
};

class CIORectangle {
public:
  CIORectangle(Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3d p3, Eigen::Vector3d p4)
      : p1_(std::move(p1)), p2_(std::move(p2)), p3_(std::move(p3)), p4_(std::move(p4)) {}
  CIORectangle(std::vector<Eigen::Vector3d> points)
      : p1_(std::move(points[0])),
        p2_(std::move(points[1])),
        p3_(std::move(points[2])),
        p4_(std::move(points[3])) {}
  Eigen::Vector3d p1() const { return p1_; }
  Eigen::Vector3d p2() const { return p2_; }
  Eigen::Vector3d p3() const { return p3_; }
  Eigen::Vector3d p4() const { return p4_; }
  Eigen::Vector3d center() const { return (p1_ + p2_ + p3_ + p4_) / 4.0; }
  Eigen::Vector3d normal() const {
    const auto n = (p2_ - p1_).cross(p4_ - p1_);
    return n.normalized();
  }

  std::vector<Eigen::Vector3d> vertices() const {
    std::vector<Eigen::Vector3d> vertices;
    vertices.push_back(p1_);
    vertices.push_back(p2_);
    vertices.push_back(p3_);
    vertices.push_back(p4_);
    return vertices;
  }

  std::vector<CIOTriangle> triangles() const {
    CIOTriangle t1(p1_, p2_, p3_);
    CIOTriangle t2(p1_, p3_, p4_);

    std::vector<CIOTriangle> triangles;
    triangles.push_back(t1);
    triangles.push_back(t2);
    return triangles;
  }

  Eigen::Vector3d intersection_ray(const CIOLine& r, double epsilon) const {
    CIOTriangle t1 = triangles()[0];
    CIOTriangle t2 = triangles()[1];
    auto result = t1.intersection_ray(r, epsilon);
    if (result != Eigen::Vector3d::Zero()) {
      return result;
    }
    return t2.intersection_ray(r, epsilon);
  }

  bool is_facing(const Eigen::Vector3d& point) const {
    Eigen::Vector3d towards_point = point - center();
    double dot_product = towards_point.dot(normal());
    return dot_product > 0;
  }

  Eigen::Vector3d project_point(const Eigen::Vector3d& point) const {
    const auto rect_normal = normal();
    return point - (rect_normal.dot(point) * rect_normal);
  }

private:
  Eigen::Vector3d p1_ = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector3d p2_ = Eigen::Vector3d(1, 0, 0);
  Eigen::Vector3d p3_ = Eigen::Vector3d(1, 1, 0);
  Eigen::Vector3d p4_ = Eigen::Vector3d(0, 1, 0);
};

class CIOEllipse {
public:
  CIOEllipse() = default;
  CIOEllipse(Eigen::Vector3d center, Eigen::Quaterniond orientation, double a, double b)
      : center_point_(std::move(center)),
        absolute_orientation_(std::move(orientation)),
        major_semi_(a),
        minor_semi_(b) {}

  double p1() const { return major_semi_; }
  double p2() const { return minor_semi_; }

  const Eigen::Vector3d center() const { return center_point_; }
  const Eigen::Quaterniond orientation() const { return absolute_orientation_; }

private:
  double major_semi_ = 0;
  double minor_semi_ = 0;
  Eigen::Vector3d center_point_ = Eigen::Vector3d::Zero();
  Eigen::Quaterniond absolute_orientation_ = Eigen::Quaterniond::Identity();
};

class CIOCuboid : public CIOObject {
public:
  enum {
    FRONT = 0,
    BACK = 1,
    LEFT = 2,
    RIGHT = 3,
    BOTTOM = 4,
    TOP = 5,
  };
  CIOCuboid() {
    for (int i = 0; i < 8; i++) {
      points_.emplace_back(Eigen::Vector3d::Zero());
    }
  }

  CIOCuboid(Eigen::Vector3d p0, Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3d p3,
            Eigen::Vector3d p4, Eigen::Vector3d p5, Eigen::Vector3d p6, Eigen::Vector3d p7) {
    points_.emplace_back(std::move(p0));
    points_.emplace_back(std::move(p1));
    points_.emplace_back(std::move(p2));
    points_.emplace_back(std::move(p3));
    points_.emplace_back(std::move(p4));
    points_.emplace_back(std::move(p5));
    points_.emplace_back(std::move(p6));
    points_.emplace_back(std::move(p7));
  }

  CIOCuboid(std::vector<Eigen::Vector3d> points) : points_(std::move(points)) {}

  CIOCuboid(const Eigen::Vector3d& center, const Eigen::Vector3d& size,
            const Eigen::Quaterniond& orientation) {
    Eigen::Vector3d p0(size.x() / 2.0, -size.y() / 2.0, -size.z() / 2.0);
    Eigen::Vector3d p1(size.x() / 2.0, size.y() / 2.0, -size.z() / 2.0);
    Eigen::Vector3d p2(size.x() / 2.0, size.y() / 2.0, size.z() / 2.0);
    Eigen::Vector3d p3(size.x() / 2.0, -size.y() / 2.0, size.z() / 2.0);

    Eigen::Vector3d p4(-size.x() / 2.0, size.y() / 2.0, -size.z() / 2.0);
    Eigen::Vector3d p5(-size.x() / 2.0, -size.y() / 2.0, -size.z() / 2.0);
    Eigen::Vector3d p6(-size.x() / 2.0, -size.y() / 2.0, size.z() / 2.0);
    Eigen::Vector3d p7(-size.x() / 2.0, size.y() / 2.0, size.z() / 2.0);

    p0 = center + orientation * p0;
    p1 = center + orientation * p1;
    p2 = center + orientation * p2;
    p3 = center + orientation * p3;

    p4 = center + orientation * p4;
    p5 = center + orientation * p5;
    p6 = center + orientation * p6;
    p7 = center + orientation * p7;

    points_.emplace_back(std::move(p0));
    points_.emplace_back(std::move(p1));
    points_.emplace_back(std::move(p2));
    points_.emplace_back(std::move(p3));
    points_.emplace_back(std::move(p4));
    points_.emplace_back(std::move(p5));
    points_.emplace_back(std::move(p6));
    points_.emplace_back(std::move(p7));
  }

  std::vector<Eigen::Vector3d> lookup_points(int face_idx) const {
    std::vector<Eigen::Vector3d> lookup;
    switch (face_idx) {
      case FRONT:
        lookup.push_back(points_[0]);
        lookup.push_back(points_[1]);
        lookup.push_back(points_[2]);
        lookup.push_back(points_[3]);
        break;
      case BACK:
        lookup.push_back(points_[4]);
        lookup.push_back(points_[5]);
        lookup.push_back(points_[6]);
        lookup.push_back(points_[7]);
        break;
      case LEFT:
        lookup.push_back(points_[1]);
        lookup.push_back(points_[4]);
        lookup.push_back(points_[7]);
        lookup.push_back(points_[2]);
        break;
      case RIGHT:
        lookup.push_back(points_[5]);
        lookup.push_back(points_[0]);
        lookup.push_back(points_[3]);
        lookup.push_back(points_[6]);
        break;
      case BOTTOM:
        lookup.push_back(points_[5]);
        lookup.push_back(points_[4]);
        lookup.push_back(points_[1]);
        lookup.push_back(points_[0]);
        break;
      case TOP:
        lookup.push_back(points_[3]);
        lookup.push_back(points_[2]);
        lookup.push_back(points_[7]);
        lookup.push_back(points_[6]);
        break;
    }
    return lookup;
  }

  std::vector<Eigen::Vector3d> vertices() const { return points_; }

  CIORectangle get_rectangle(int face_idx) const { return CIORectangle(lookup_points(face_idx)); }

  Eigen::Vector3d center() const {
    Eigen::Vector3d point_sum = points_[0];
    for (int i = 1; i < 8; i++) {
      point_sum += points_[i];
    }
    return point_sum / 8.0;
  }

  std::vector<Eigen::Vector3d> intersection_ray(const CIOLine& r, double epsilon) const {
    std::vector<Eigen::Vector3d> ret;
    for (int i = 0; i < 6; i++) {
      CIORectangle side = get_rectangle(i);
      auto side_intersect = side.intersection_ray(r, epsilon);
      if (side_intersect != Eigen::Vector3d::Zero()) {
        ret.push_back(side_intersect);
      }
    }
    return ret;
  }

  Eigen::Vector3d project_point(const Eigen::Vector3d& point) const override {
    static constexpr float k = 1.e4;
    static constexpr int num_lines = 6;

    // Initialize p_nearest as a zero matrix
    Eigen::MatrixXd p_nearest = Eigen::MatrixXd::Zero(num_lines, 3);
    for (auto j = 0; j < num_lines; ++j) {
      p_nearest.row(j) = get_rectangle(j).project_point(point);
    }

    // Transpose the point and tile it
    const auto p_mat = point.replicate(num_lines, 1);

    // Create ones vector
    const auto ones_vec = Eigen::VectorXd::Ones(num_lines);

    // Calculate nu
    const auto distances = (p_mat - p_nearest).rowwise().squaredNorm();
    Eigen::VectorXd nu = ones_vec.array() / (ones_vec.array() + distances.array() * k);

    // Normalize nu
    nu /= nu.sum();

    // Tile nu for broadcasting
    Eigen::MatrixXd nu_tiled = nu.replicate(1, 3);

    // Calculate closest point
    const auto closest_point = (nu_tiled.array() * p_nearest.array()).colwise().sum();
    return closest_point;
  }

private:
  std::vector<Eigen::Vector3d> points_;
};

class CIOCylinder : public CIOObject {
public:
  enum {
    BOTTOM = 0,
    TOP = 1,
  };
  CIOCylinder(Eigen::Vector3d center, double radius, double height, Eigen::Quaterniond orientation)
      : center_point_(std::move(center)),
        radius_(radius),
        height_(height),
        absolute_orientation_(std::move(orientation)) {}

  Eigen::Vector3d center() const { return center_point_; }
  Eigen::Quaterniond orientation() const { return absolute_orientation_; }

  double r() const { return radius_; }
  double h() const { return height_; }

  CIOEllipse get_cap(int index) const {
    CIOEllipse e;
    Eigen::Vector3d ellipse_center;
    switch (index) {
      case BOTTOM:
        ellipse_center = center() - orientation() * (Eigen::Vector3d::UnitZ() * (h() / 2.0));
        e = CIOEllipse(ellipse_center, orientation(), r(), r());
        break;
      case TOP:
        ellipse_center = center() + orientation() * (Eigen::Vector3d::UnitZ() * (h() / 2.0));
        e = CIOEllipse(ellipse_center, orientation(), r(), r());
        break;
    }
    return e;
  }

private:
  Eigen::Vector3d center_point_ = Eigen::Vector3d::Zero();
  double radius_ = 0;
  double height_ = 0;
  Eigen::Quaterniond absolute_orientation_ = Eigen::Quaterniond::Identity();
};

class CIOCone : public CIOObject {
public:
  CIOCone(Eigen::Vector3d origin_point, double angle, double height,
          const Eigen::Vector3d& absolute_direction)
      : origin_point_(std::move(origin_point)),
        angle_(angle),
        height_(height),
        absolute_direction_(absolute_direction.normalized()) {}
  Eigen::Vector3d origin() const { return origin_point_; }
  Eigen::Vector3d direction() const { return absolute_direction_; }
  Eigen::Vector3d center() const { return origin() + (0.5 * h()) * direction(); }

  double theta() const { return angle_; }
  double h() const { return height_; }

  CIOEllipse get_cap() const {
    Eigen::Vector3d ellipse_center = origin() + direction() * h();
    Eigen::Quaterniond ellipse_orientation =
        Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), direction());
    double cap_radius = std::tan(theta()) * h();
    CIOEllipse e(ellipse_center, ellipse_orientation, cap_radius, cap_radius);
    return e;
  }

  Eigen::Vector3d project_point(const Eigen::Vector3d& point) const override {
    Eigen::Vector3d point_vec = point - origin();
    double point_axis_angle = acos((point_vec.dot(direction())) / (point_vec.norm() * direction().norm()));

    /* Eigen::Vector3d axis_projection = this->cone_axis_projector * point_vec + origin(); */

    Eigen::Vector3d axis_rot = direction().cross(point_vec);
    axis_rot.normalize();

    Eigen::AngleAxis<double> my_quat(this->angle_ - point_axis_angle, axis_rot);

    Eigen::Vector3d point_on_cone = my_quat * point_vec + origin();

    Eigen::Vector3d vec_point_on_cone = point_on_cone - origin();
    vec_point_on_cone.normalize();

    double beta = this->angle_ - point_axis_angle;

    if (point_axis_angle < this->angle_) {
      return origin() + vec_point_on_cone * cos(beta) * point_vec.norm();
    } else if ((point_axis_angle >= this->angle_) &&
               (point_axis_angle - this->angle_) <= M_PI_2) {  // TODO: is this condition correct?
      return origin() + vec_point_on_cone * cos(point_axis_angle - this->angle_) * point_vec.norm();
    } else {
      return Eigen::Vector3d::Zero();
    }
  }

private:
  Eigen::Vector3d origin_point_ = Eigen::Vector3d::Zero();
  double angle_;
  double height_;
  Eigen::Vector3d absolute_direction_ = Eigen::Vector3d::Zero();
};

class CIOSphere : public CIOObject {
public:
  CIOSphere() = default;
  CIOSphere(int id, double radius, CIOPose pose, const Eigen::Vector3d& vel, double step_size = 0.5)
      : CIOObject(id, std::move(pose), CIOVelocity{.linear_vel = vel}, step_size), radius_(radius) {}

  // Projects the given point onto the surface of this object
  Eigen::Vector3d project_point(const Eigen::Vector3d& point) const override {
    return pose_.position() + (radius_ * get_surface_normal(point));
  }

  double r() const { return radius_; }

private:
  double radius_ = 0;
};
using CIOSpherePtr = std::shared_ptr<CIOSphere>;
#else
class CIOLine : public CIOObject {
public:
  CIOLine() = default;
  CIOLine(int id, double length, const Eigen::Vector3d& pos,
          const Eigen::Vector3d& vel = Eigen::Vector3d::Zero(), double step_size = 0.5)
      : CIOObject(id, CIOPose(pos), CIOVelocity{.linear_vel = vel}, step_size), length_(length) {}

  std::vector<Eigen::Vector3d> discretize() {
    int N_points = floor(length_ / step_size_) + 1;
    std::vector<Eigen::Vector3d> points(N_points);
    const auto [p0, p1] = get_endpoints();
    points[0] = p0;
    points[N_points - 1] = p1;
    const auto direction = (p1 - p0).normalized();
    for (auto i = 1; i < N_points - 1; ++i) {
      const auto [pi_0, pi_1] = get_step_size_points(points[i] + step_size_ * direction);
      points[i] = pi_0;
      points[i - 1] = pi_1;
    }
    return points;
  }

  std::pair<Eigen::Vector3d, Eigen::Vector3d> get_step_size_points(const Eigen::Vector3d& p) {
    const auto p_direction = (Eigen::Affine3d(pose_.orientation()) * p).normalized();
    return {p - 0.5 * step_size_ * p_direction, p + 0.5 * step_size_ * p_direction};
  }

  std::pair<Eigen::Vector3d, Eigen::Vector3d> get_endpoints() {
    const auto p0 = pose_.position();
    const auto p0_direction = (Eigen::Affine3d(pose_.orientation()) * p0).normalized();
    return {p0 - 0.5 * length_ * p0_direction, p0 + 0.5 * length_ * p0_direction};
  }

  // project a given point onto this line (or endpoint)
  // https://gamedev.stackexchange.com/questions/72528/how-can-i-project-a-3d-point-onto-a-3d-line
  Eigen::Vector3d project_point(const Eigen::Vector3d& p) const override {
    const auto [a, b] = get_endpoints();
    const auto ap = p - a;
    const auto ab = b - a;
    return a + ap.dot(ab) / ab.dot(ab) * ab;
  }
  double length_;
};

class CIOCuboid : public CIOObject {
public:
  CIOCuboid() = default;
  CIOCuboid(int id, const Eigen::Vector3d& size, const Eigen::Vector3d& pos,
            const Eigen::Vector3d& vel = Eigen::Vector3d::Zero(), double step_size = 0.5)
      : CIOObject(id, CIOPose(pos), CIOVelocity{.linear_vel = vel}, step_size),
        width_(size[0]),
        length_(size[1]),
        height_(size[2]) {
    // Cuboid rectangles are made up of 12 line objects
    // lines_ = make_lines()
  }

  std::vector<CIOLine> edges;
  double width_;
  double length_;
  double height_;
};
#endif

using CIOFinger = CIOSphere;
using CIOFingerPtr = std::shared_ptr<CIOFinger>;
using CIOContactMap = std::map<CIOObjectPtr, std::vector<CIOContact>>;

// NOTE: CONSIDER USING [mj_geomDistance] between finger tip sites & object sites
class CIOWorld {
public:
  using CIOWorldPtr = std::shared_ptr<CIOWorld>;
  CIOWorld() = default;
  CIOWorld(CIOObjectPtr manip_obj, std::vector<CIOFingerPtr> fingers, CIOContactMap contact_states_,
           std::function<void()> traj_func = nullptr)
      : manip_obj_(std::move(manip_obj)),
        fingers_(std::move(fingers)),
        contact_states_(std::move(contact_states_)),
        traj_func_(std::move(traj_func)) {}

  void set_dynamics(int obj_idx, const CIOPose& pose, const CIOVelocity& vel, const CIOAcceleration& acc) {
    auto objects = get_all_objects();
    objects[obj_idx]->set_dynamics(pose, vel, acc);
  }

  void set_contact_state(const CIOObjectPtr& obj, const Eigen::Vector3d& f, const Eigen::Vector3d& ro,
                         double c) {
    if (contact_states_.contains(obj)) {
      contact_states_[obj] = {CIOContact{.f = f, .ro = ro, .c = c}};
    }
  }

  void set_e_vars(const CIOConfig& config, const CIOWorldPtr& world_tm1 = nullptr) {
    const auto obj_pose = manip_obj_->pose();
    for (auto& [object, contact_list] : contact_states_) {
      for (auto i = 0; i < contact_list.size(); ++i) {
        auto& contact_i = contact_list[i];
        const auto r = obj_pose.position() + contact_i.ro;
        const auto ci = object->id();
        contact_i.pi_H_ = object->project_point(r);
        contact_i.pi_O_ = manip_obj_->project_point(r);
        contact_i.e_H_ = contact_i.pi_H_ - r;
        contact_i.e_O_ = contact_i.pi_O_ - r;

        if (world_tm1) {
          const auto world_tm1_contact = world_tm1->get_contact(object, i);
          contact_i.e_dot_H_ = CIOUtils::calc_derivative(contact_i.e_H_, world_tm1_contact.e_H_, config.delT);
          contact_i.e_dot_O_ = CIOUtils::calc_derivative(contact_i.e_O_, world_tm1_contact.e_O_, config.delT);
        } else {
          contact_i.e_dot_H_ = Eigen::Vector3d::Zero();
          contact_i.e_dot_O_ = Eigen::Vector3d::Zero();
        }
      }
    }
  }

  double l_contacts() const {
    double cost = 0;
    for (const auto& [contact_obj, contact_list] : contact_states_) {
      for (const auto& contact : contact_list) {
        cost += contact.c * (pow(contact.e_O_.norm(), 2) + pow(contact.e_H_.norm(), 2) +
                             pow(contact.e_dot_O_.norm(), 2) + pow(contact.e_dot_H_.norm(), 2));
      }
    }
    return cost;
  }

  // Kinematics cost: 1) limits on finger and arm joint angles
  //                  2) distance from fingertips to palms limit
  //                  3) collisions between fingers
  double l_kinematics() const {
    double cost = 0;
    const auto all_objects = get_all_objects();
    int i = 0;
    while (i < all_objects.size()) {
      for (const auto& obj : all_objects) {
        cost += all_objects[i]->check_collisions(*obj);
        i++;
      }
    }
    return cost;
  }

  double l_physics() const {
    // calculate sum of forces on object
    Eigen::Vector3d f_tot = Eigen::Vector3d::Zero();
    for (const auto& [contact_obj, contact_list] : contact_states_) {
      for (const auto& contact : contact_list) {
        f_tot += contact.c * contact.f;
      }
    }
    const auto oa = manip_obj_->acc().linear_acc;
    double newton_cost = (f_tot - config_.mass * oa).norm();
    newton_cost = pow(newton_cost, 2);

    // Calculate force regularization cost
    double force_reg_cost = 0.0;
    for (const auto& [contact_obj, contact_list] : contact_states_) {
      for (const auto& contact : contact_list) {
        force_reg_cost += pow(contact.f.norm(), 2);
      }
    }
    force_reg_cost *= config_.lamb;

    // Calculate L_cone
    double cone_cost = 0.0;
    for (const auto& [contact_obj, contact_list] : contact_states_) {
      for (const auto& contact : contact_list) {
        const auto n = contact_obj->get_surface_normal(contact.pi_H_);
        const auto f_n = contact.f.normalized();
        double angle = acos(f_n.dot(n));
        cone_cost += pow(std::max(angle - atan(config_.mu), 0.0), 2);
      }
    }

    return force_reg_cost + newton_cost + cone_cost;
  }

  double l_task(int t, const std::vector<CIOGoal>& goals) {
    // Task constraint: get object to desired position
    int I = (t == (config_.T_steps() - 1)) ? 1 : 0;
    double task_cost = 0.0;

    for (const auto& goal : goals) {
      // Position
      const auto obj_pos = manip_obj_->pose().position();
      const auto goal_pos = goal.pose.position();
      task_cost += I * (obj_pos - goal_pos).squaredNorm();

      // Velocity
      const auto obj_vel = manip_obj_->vel().linear_vel;
      const auto goal_vel = goal.vel.linear_vel;
      task_cost += I * (obj_vel - goal_vel).squaredNorm();
    }

    // Small acceleration constraint
    double accel_cost = 0.0;
    for (const auto& obj : get_all_objects()) {
      accel_cost += pow(obj->acc().linear_acc.norm(), 2);
    }
    accel_cost *= config_.lamb;

    return accel_cost + task_cost;
  }

  double total_cost(int stage_idx, const std::vector<CIOGoal>& goals) {
    const auto dynamic_worlds = dynamic_traj();
    double ci = 0.0, phys = 0.0, kinem = 0.0, task = 0.0;
    for (auto t = 0; t < dynamic_worlds.size(); ++t) {
      const auto& world_t = dynamic_worlds[t];
      const auto stage_weight = config_.stage_weights[stage_idx];
      ci += stage_weight.w_CI * world_t->l_contacts();
      phys += stage_weight.w_physics * world_t->l_physics();
      kinem += stage_weight.w_kinematics * world_t->l_kinematics();
      task += stage_weight.w_task * world_t->l_task(t, goals);
    }
    return ci + phys + kinem + task;
  }

  std::vector<CIOObjectPtr> get_all_objects() const {
    std::vector<CIOObjectPtr> objects = {manip_obj_};
    for (const auto& finger : fingers_) {
      objects.push_back(finger);
    }
    return objects;
  }

  std::vector<CIOObservation> get_observations() const {
    std::vector<CIOObservation> s;
    for (const auto& obj : get_all_objects()) {
      s.push_back(CIOObservation{.obj_id = obj->id(), .pose = obj->pose(), .vel = obj->vel()});
    }
    for (const auto& [obj, contact_list] : contact_states_) {
      for (const auto& contact : contact_list) {
        s.push_back(
            CIOObservation{.obj_id = obj->id(), .pose = obj->pose(), .vel = obj->vel(), .contact = contact});
      }
    }
    return s;
  }

  std::vector<CIOObservation> calc_obj_dynamics() const {
    const std::vector<CIOObservation> observations = get_observations();

    // Make splines from [pos_traj_K, vel_traj_K]
    std::vector<Eigen::Vector3d> pos_traj_K(config_.K + 1);
    std::vector<Eigen::Vector3d> vel_traj_K(config_.K + 1);
    auto splines = CIOUtils::create_splines(pos_traj_K, vel_traj_K, config_);
    for (auto& obs : observations) {
      pos_traj_K.push_back(obs.pose.position());
      vel_traj_K.push_back(obs.vel.linear_vel);
    }

    std::vector<Eigen::Vector3d> pos_traj_T;
    int k = 0;
    std::vector<double> times = CIOUtils::linspace<double>(0., config_.T_final(), config_.T_steps() + 1);
    for (const auto t : times) {
      if (fmod(t, config_.delT_phase) != 0.) {
        pos_traj_T.push_back(pos_traj_K[k]);
        k++;
      } else {
        pos_traj_T.push_back(
            Eigen::Vector3d{splines[0](t).value(), splines[1](t).value(), splines[2](t).value()});
      }
    }

    std::vector<Eigen::Vector3d> vel_traj_T;
    for (auto t = 1; t < config_.T_steps() + 1; ++t) {
      vel_traj_T.push_back(CIOUtils::calc_derivative(pos_traj_T[t], pos_traj_T[t - 1], config_.delT));
    }

    std::vector<Eigen::Vector3d> acc_traj_T;
    for (auto t = 1; t < config_.T_steps() + 1; ++t) {
      vel_traj_T.push_back(CIOUtils::calc_derivative(vel_traj_T[t], vel_traj_T[t - 1], config_.delT));
    }

    std::vector<CIOObservation> out_observations;
    for (auto t = 1; t < config_.T_steps() + 1; ++t) {
      out_observations.push_back(CIOObservation{.pose = CIOPose(pos_traj_T[t]),
                                                .vel = CIOVelocity{.linear_vel = vel_traj_T[t]},
                                                .acc = CIOAcceleration{.linear_acc = acc_traj_T[t]}});
    }
    return out_observations;
  }

  std::vector<CIOContact> get_object_contacts(const CIOObjectPtr& object) const {
    return contact_states_.contains(object) ? contact_states_.at(object) : std::vector<CIOContact>{};
  }

  CIOContact get_contact(const CIOObjectPtr& object, int contact_idx) const {
    auto contacts = get_object_contacts(object);
    return contacts.empty() ? CIOContact() : contacts[contact_idx];
  }

  std::vector<CIOContact> get_smooth_contacts(const CIOObjectPtr& object) const {
    auto contacts = get_object_contacts(object);
    if (contacts.empty()) {
      return {};
    }

    // CIOTODO
    auto s0 = get_observations();
    auto trajs = stationary_trajs();
    std::vector<CIOContact> contact_traj_K;
    for (auto k = 0; k < config_.K; ++k) {
      for (const auto& s0_i : s0) {
        contact_traj_K.push_back(s0_i.contact);
      }
      for (const auto& traj : trajs) {
        for (const auto& traj_i : traj) {
          contact_traj_K.push_back(traj_i.contact);
        }
      }
    }

    std::vector<CIOContact> contact_traj_T;
    for (auto k = 1; k < contact_traj_K.size() + 1; ++k) {
      auto contact_left = contact_traj_K[k - 1];
      auto contact_right = contact_traj_K[k];

      // interpolate the contact forces (linear)
      auto f_traj_T =
          CIOUtils::linspace_vectors(contact_left.f, contact_right.f, config_.steps_per_phase() + 1);
      auto ro_traj_T =
          CIOUtils::linspace_vectors(contact_left.ro, contact_right.ro, config_.steps_per_phase() + 1);
      for (auto t = (k - 1) * config_.steps_per_phase(); t < k * config_.steps_per_phase() + 1; ++t) {
        contact_traj_T.push_back(CIOContact{.f = f_traj_T[t], .ro = ro_traj_T[t], .c = contact_traj_K[k].c});
      }
    }

    return contact_traj_T;
  }

  std::vector<std::vector<CIOObservation>> stationary_trajs() const {
    std::vector<std::vector<CIOObservation>> trajs;
    for (auto k = 0; k < config_.K; ++k) {
      trajs.push_back(get_observations());
    }
    return trajs;
  }

  std::vector<CIOWorldPtr> dynamic_traj() const {
    // Dynamics
    std::vector<std::vector<CIOObservation>> all_dyn_info;
    for (auto i = 0; i < get_all_objects().size(); ++i) {
      all_dyn_info.push_back(calc_obj_dynamics());
    }

    // Contacts
    CIOContactMap all_contact_info;
    for (const auto& [contact_obj, contact_list] : contact_states_) {
      all_contact_info.insert_or_assign(contact_obj, get_smooth_contacts(contact_obj));
    }

    // Fill into list of new worlds
    std::vector<CIOWorldPtr> worlds;
    for (int t = 0; t < config_.T_steps() + 1; ++t) {
      CIOWorldPtr world_t = std::make_shared<CIOWorld>(*this);

      // World's objs dynamics
      for (auto i = 0; i < world_t->get_all_objects().size(); ++i) {
        world_t->set_dynamics(i, all_dyn_info[i][0].pose, all_dyn_info[i][0].vel, all_dyn_info[i][0].acc);
      }

      // World's objs contact
      for (const auto& [contact_obj, contact_list] : contact_states_) {
        const auto contact = all_contact_info.at(contact_obj)[0];
        world_t->set_contact_state(contact_obj, contact.f, contact.ro, contact.c);
      }

      world_t->set_e_vars(config_, (t == 0) ? nullptr : worlds[t - 1]);
      worlds.emplace_back(std::move(world_t));
    }
    return worlds;
  }

  CIOObjectPtr manip_obj_ = nullptr;
  std::vector<CIOFingerPtr> fingers_;
  CIOContactMap contact_states_;
  std::function<void()> traj_func_;
  CIOConfig config_;

  // each dynamic object has a 3D pose and vel and each contact surface has 7 associated vars
  int N() const { return contact_states_.size(); }
  int len_s() const { return int(6 * get_all_objects().size() + N() * 7); }
  // add accelerations of dynamic objects
  int len_s_aug() const { return int(len_s() + 3. * get_all_objects().size()); }
  int len_S() const { return int(len_s() * config_.K); }
  int len_S_aug() const { return int(len_s_aug() * config_.K * config_.steps_per_phase()); }
};
using CIOWorldPtr = CIOWorld::CIOWorldPtr;
