#pragma once

#include <mujoco/mujoco.h>

#include <chrono>
#include <map>
#include <memory>
#include <random>

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

// mjpc
#include "mjpc/casadi/casadi_common.h"
#include "mjpc/task.h"

#define CIO_USE_LBFGSB (1)
#define CIO_USE_EXT_OBJ_WRENCH (0)

struct CIOPose {
  CIOPose() = default;
  explicit CIOPose(const Eigen::Vector3d& position) : trans(position) {}
  explicit CIOPose(const Eigen::Quaterniond& orientation) : quat(orientation) {}
  CIOPose(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation)
      : trans(position), quat(orientation) {}
  Eigen::Vector3d position() const { return trans.vector(); }
  Eigen::Quaterniond orientation() const { return quat; }
  Eigen::Vector3d rpy() const {
    Eigen::Vector3d eulerAngles = quat.toRotationMatrix().eulerAngles(2, 1, 0);  // ZYX order
    return {eulerAngles(2), eulerAngles(1), eulerAngles(0)};
  }

  Eigen::Translation3d trans;
  Eigen::Quaterniond quat;
  void add_noise();
  constexpr int size() const { return size_byte() / sizeof(double); }
  constexpr int size_byte() const { return sizeof(trans) + sizeof(quat); }
  std::vector<double> data() const {
    std::vector<double> out(size());
    std::memcpy(out.data(), trans.translation().data(), sizeof(trans));
    std::memcpy(out.data() + (sizeof(trans) / sizeof(double)), quat.coeffs().data(), sizeof(quat));
    return out;
  }

  void from_data(const double* data) {
    std::memcpy(trans.translation().data(), data, sizeof(trans));
    std::memcpy(quat.coeffs().data(), data + (sizeof(trans) / sizeof(double)), sizeof(quat));
  }
};

struct CIOVelocity {
  Eigen::Vector3d linear_vel = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_vel = Eigen::Vector3d::Zero();
  void add_noise();
  constexpr int size() const { return size_byte() / sizeof(double); }
  constexpr int size_byte() const { return sizeof(linear_vel) + sizeof(angular_vel); }
  std::vector<double> data() const {
    std::vector<double> out(size());
    std::memcpy(out.data(), linear_vel.data(), sizeof(linear_vel));
    std::memcpy(out.data() + (sizeof(linear_vel) / sizeof(double)), angular_vel.data(), sizeof(angular_vel));
    return out;
  }
  void from_data(const double* data) {
    std::memcpy(linear_vel.data(), data, sizeof(linear_vel));
    std::memcpy(angular_vel.data(), data + (sizeof(linear_vel) / sizeof(double)), sizeof(angular_vel));
  }
};

struct CIOAcceleration {
  Eigen::Vector3d linear_acc = Eigen::Vector3d::Zero();
  Eigen::Vector3d angular_acc = Eigen::Vector3d::Zero();
  void add_noise();
  constexpr int size() const { return size_byte() / sizeof(double); }
  constexpr int size_byte() const { return sizeof(linear_acc) + sizeof(angular_acc); }
  std::vector<double> data() const {
    std::vector<double> out(size());
    std::memcpy(out.data(), linear_acc.data(), sizeof(linear_acc));
    std::memcpy(out.data() + (sizeof(linear_acc) / sizeof(double)), angular_acc.data(), sizeof(angular_acc));
    return out;
  }
  void from_data(const double* data) {
    std::memcpy(linear_acc.data(), data, sizeof(linear_acc));
    std::memcpy(angular_acc.data(), data + (sizeof(linear_acc) / sizeof(double)), sizeof(angular_acc));
  }
};

struct CIOGoal {
  CIOPose pose;
  CIOVelocity vel;
  // CIOAcceleration acc;
};

struct CIOContact {
  int id = 0;
  // Contact force
  Eigen::Vector3d f = Eigen::Vector3d::Zero();
  // Position of applied force in the frame of the manipulated object
  Eigen::Vector3d ro = Eigen::Vector3d::Zero();
  // [0,1]: Probability of being in contact
  // double c = 0;
  // Distance between nearest points; neg: penetration
  double dist = 0;
  double dist_dot = 0;

  // Position of applied force in world frame
  Eigen::Vector3d r = Eigen::Vector3d::Zero();
  // Projection of applied force onto object
  Eigen::Vector3d pi_O_ = Eigen::Vector3d::Zero();
  // Projection of applied force onto hand
  Eigen::Vector3d pi_H_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d e_O_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d e_H_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d e_dot_O_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d e_dot_H_ = Eigen::Vector3d::Zero();
  bool empty() const { return f.isZero() && ro.isZero(); }
  void add_noise();
  constexpr int size() const { return size_byte() / sizeof(double); }
  constexpr int size_byte() const { return sizeof(f) + sizeof(ro); }
  std::vector<double> data() const {
    std::vector<double> out(size());
    std::memcpy(out.data(), f.data(), sizeof(f));
    std::memcpy(out.data() + (sizeof(f) / sizeof(double)), ro.data(), sizeof(ro));
    return out;
  }
  void from_data(const double* data) {
    std::memcpy(f.data(), data, sizeof(f));
    std::memcpy(ro.data(), data + (sizeof(f) / sizeof(double)), sizeof(ro));
  }
};

struct CIOObservation {
  int obj_id = -1;
  CIOPose pose;
  CIOVelocity vel;
  CIOAcceleration acc;
  std::vector<CIOContact> contacts;

  static const int pose_size;
  static const int vel_size;
  static const int acc_size;
  static const int pose_vel_size;
  static const int pose_vel_acc_size;
  static const int contact_size;
  void add_noise();
  std::vector<double> data() const {
    std::vector<double> out(pose_size + vel_size + acc_size + contacts.size() * contact_size);
    std::memcpy(out.data(), pose.data().data(), pose_size * sizeof(double));
    std::memcpy(out.data() + pose_size, vel.data().data(), vel_size * sizeof(double));
    std::memcpy(out.data() + pose_vel_size, acc.data().data(), acc_size * sizeof(double));
    for (auto i = 0; i < contacts.size(); ++i) {
      const auto& contact_i = contacts[i];
      std::memcpy(out.data() + pose_vel_acc_size + i * contact_size, contact_i.data().data(),
                  contact_size * sizeof(double));
    }
    return out;
  }
  void from_data(const double* data, int elem_num) {
    pose.from_data(data);
    vel.from_data(data + pose_size);
    acc.from_data(data + pose_vel_size);
    const auto contacts_num = (elem_num - pose_vel_acc_size) / contact_size;
    contacts.resize(contacts_num);
    for (auto i = 0; i < contacts_num; ++i) {
      contacts[i].from_data(data + pose_vel_acc_size + i * contact_size);
    }
  }
  void from_data(const std::vector<double>& data_vec) { from_data(data_vec.data(), (int)data_vec.size()); }
};

struct CIOStageWeight {
  double w_CI = 0;
  double w_physics = 0;
  double w_kinematics = 0;
  double w_task = 0;
};

struct CIOConfig {
  int K = 10;
  double delT = 0.001;
  double delT_phase = 0.5;
  double mass = 1.0;
  double mu = 0.9;      // Friction coefficient
  double lamb = 0.001;  // Regularization parameter

  std::vector<CIOStageWeight> stage_weights = {
      CIOStageWeight{.w_CI = 0.1, .w_physics = 0.1, .w_kinematics = 0.0, .w_task = 1.0},
      CIOStageWeight{.w_CI = 10.0, .w_physics = 1.0, .w_kinematics = 0.0, .w_task = 10.0}};
  std::function<void()> init_traj = nullptr;

  int steps_per_phase() const { return int(delT_phase / delT); }
  int T_steps() const { return K * steps_per_phase(); }
  double T_final() const { return K * delT_phase; }
};

class CIOObject {
public:
  CIOObject() = default;
  CIOObject(int body_id, int geom_id, double step_size = 0.5)
      : body_id_(body_id), geom_id_(geom_id), step_size_(step_size) {}

  void set_mj_info(const mjModel* model, const mjData* data, const mjpc::Task* task) {
    mj_model_ = model;
    mj_data_ = data;
    mj_task_ = task;
  }
  int id() const { return body_id_; }
  int geom_id() const { return geom_id_; }
  double mass() const { return mj_task_->QueryBodyMass(id()); }
  CIOPose pose() const {
    return CIOPose(Eigen::Vector3d(mj_task_->QueryBodyPos(id())),
                   Eigen::Quaterniond(mj_task_->QueryBodyQuat(id())));
  }
  CIOVelocity vel() const {
    return CIOVelocity{.linear_vel = Eigen::Vector3d(mj_task_->QueryBodyVel(id())),
                       .angular_vel = Eigen::Vector3d(mj_task_->QueryBodyVel(id(), false))};
  }

  CIOAcceleration acc() const {
    return CIOAcceleration{.linear_acc = Eigen::Vector3d(mj_task_->QueryBodyAcc(id())),
                           .angular_acc = Eigen::Vector3d(mj_task_->QueryBodyAcc(id(), false))};
  }

  Eigen::Vector3d get_surface_normal(const Eigen::Vector3d& point) const {
    return (point - pose().position()).normalized();
  }

  virtual Eigen::Vector3d project_point(const Eigen::Vector3d& point) const {
    return Eigen::Vector3d::Zero();
  }
  virtual void discretize() {}
  virtual bool check_inside(const Eigen::Vector3d& point) { return false; }

protected:
  const mjModel* mj_model_ = nullptr;
  const mjData* mj_data_ = nullptr;
  const mjpc::Task* mj_task_ = nullptr;
  int body_id_ = 0;
  int geom_id_ = 0;
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
  Eigen::Vector3d center() const { return 0.25 * (p1_ + p2_ + p3_ + p4_); }
  Eigen::Vector3d normal() const { return ((p2_ - p1_).cross(p4_ - p1_)).normalized(); }

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
  CIOCuboid(int body_id, int geom_id) : CIOObject(body_id, geom_id) {
    for (int i = 0; i < 8; i++) {
      points_.emplace_back(Eigen::Vector3d::Zero());
    }
  }

  CIOCuboid(int body_id, int geom_id, Eigen::Vector3d p0, Eigen::Vector3d p1, Eigen::Vector3d p2,
            Eigen::Vector3d p3, Eigen::Vector3d p4, Eigen::Vector3d p5, Eigen::Vector3d p6,
            Eigen::Vector3d p7)
      : CIOObject(body_id, geom_id) {
    points_.emplace_back(std::move(p0));
    points_.emplace_back(std::move(p1));
    points_.emplace_back(std::move(p2));
    points_.emplace_back(std::move(p3));
    points_.emplace_back(std::move(p4));
    points_.emplace_back(std::move(p5));
    points_.emplace_back(std::move(p6));
    points_.emplace_back(std::move(p7));
  }

  CIOCuboid(int body_id, int geom_id, std::vector<Eigen::Vector3d> points)
      : CIOObject(body_id, geom_id), points_(std::move(points)) {}

  CIOCuboid(int body_id, int geom_id, const Eigen::Vector3d& center, const Eigen::Vector3d& radius,
            const Eigen::Quaterniond& orientation)
      : CIOObject(body_id, geom_id) {
    set_points(center, radius, orientation);
  }

  void set_points(const Eigen::Vector3d& center, const Eigen::Vector3d& radius,
                  const Eigen::Quaterniond& orientation) {
    Eigen::Vector3d p0(radius.x(), -radius.y(), -radius.z());
    Eigen::Vector3d p1(radius.x(), radius.y(), -radius.z());
    Eigen::Vector3d p2(radius.x(), radius.y(), radius.z());
    Eigen::Vector3d p3(radius.x(), -radius.y(), radius.z());

    Eigen::Vector3d p4(-radius.x(), radius.y(), -radius.z());
    Eigen::Vector3d p5(-radius.x(), -radius.y(), -radius.z());
    Eigen::Vector3d p6(-radius.x(), -radius.y(), radius.z());
    Eigen::Vector3d p7(-radius.x(), radius.y(), radius.z());

    p0 = center + orientation * p0;
    p1 = center + orientation * p1;
    p2 = center + orientation * p2;
    p3 = center + orientation * p3;

    p4 = center + orientation * p4;
    p5 = center + orientation * p5;
    p6 = center + orientation * p6;
    p7 = center + orientation * p7;

    points_.clear();
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

  CIORectangle get_rectangle(int face_idx) const { return {lookup_points(face_idx)}; }

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
    static constexpr int num_faces = 6;

    // Initialize p_nearest as a zero matrix
    Eigen::MatrixXd p_nearest = Eigen::MatrixXd::Zero(num_faces, 3);
    for (auto j = 0; j < num_faces; ++j) {
      p_nearest.row(j) = get_rectangle(j).project_point(point);
    }

    // Transpose the point and tile it
    Eigen::MatrixXd p_mat = Eigen::MatrixXd::Zero(num_faces, 3);
    for (auto j = 0; j < num_faces; ++j) {
      p_mat.row(j) = point;
    }

    // Create ones vector
    const auto ones_vec = Eigen::VectorXd::Ones(num_faces);

    // Calculate nu, using a softmin instead of a hardmin to make function smooth
    // https://research.cs.wisc.edu/zhu/space2/TTP2/advanced_image_selection/bin/toolbox/doc/classify/softmin.html
    // Let d be a vector.  Then the softmin of d is defined as:
    // s = exp(-d / sigma ^ 2) / sum(exp(-d / sigma ^ 2))
    // The softmin is a way of taking a dissimilarity(distance) vector d and
    // converting it to a similarity vector s,
    // such that sum(s) == 1.
    const auto d = (p_mat - p_nearest).rowwise().squaredNorm();
#if 1
    const auto sigma = k;
    Eigen::VectorXd nu = (-d.array() / (sigma * sigma)).exp();
#else
    Eigen::VectorXd nu = ones_vec.array() / (ones_vec.array() + d.array() * k);
#endif

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
        ellipse_center = center() - orientation() * (0.5 * h() * Eigen::Vector3d::UnitZ());
        e = CIOEllipse(ellipse_center, orientation(), r(), r());
        break;
      case TOP:
        ellipse_center = center() + orientation() * (0.5 * h() * Eigen::Vector3d::UnitZ());
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
  double angle_ = 0;
  double height_ = 0;
  Eigen::Vector3d absolute_direction_ = Eigen::Vector3d::Zero();
};

class CIOSphere : public CIOObject {
public:
  CIOSphere() = default;
  CIOSphere(int body_id, int geom_id, double radius, double step_size = 0.5)
      : CIOObject(body_id, geom_id, step_size), radius_(radius) {}

  // Projects the given point onto the surface of this object
  Eigen::Vector3d project_point(const Eigen::Vector3d& point) const override {
    return pose().position() + (radius_ * get_surface_normal(point));
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
    const auto p_direction = (Eigen::Affine3d(pose().orientation()) * p).normalized();
    return {p - 0.5 * step_size_ * p_direction, p + 0.5 * step_size_ * p_direction};
  }

  std::pair<Eigen::Vector3d, Eigen::Vector3d> get_endpoints() {
    const auto p0 = pose().position();
    const auto p0_direction = (Eigen::Affine3d(pose().orientation()) * p0).normalized();
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
  CIOCuboid(int body_id, int geom_id, const Eigen::Vector3d& size, const Eigen::Vector3d& pos,
            const Eigen::Vector3d& vel = Eigen::Vector3d::Zero(), double step_size = 0.5)
      : CIOObject(body_id, geom_id, CIOPose(pos), CIOVelocity{.linear_vel = vel}, step_size),
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
