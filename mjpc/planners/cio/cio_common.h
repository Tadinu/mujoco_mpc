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

#define CIO_USE_LBFGSB (0)
#define CIO_USE_EXT_OBJ_WRENCH (0)

struct CIOPose {
  CIOPose() = default;
  CIOPose(const CaSX& position, const CaSX& orientation = mjpc_casadi::CASX_ORIENTATION_ZERO)
      : trans(position), quat(orientation) {}
  CaSX position() const { return trans; }
  CaSX orientation() const { return quat; }
  Eigen::Vector3d rpy() const {
    const Eigen::Quaterniond eigen_quat = {(double)quat(0).scalar(), (double)quat(1).scalar(),
                                           (double)quat(2).scalar(), (double)quat(3).scalar()};
    Eigen::Vector3d eulerAngles = eigen_quat.toRotationMatrix().eulerAngles(2, 1, 0);  // ZYX order
    return {eulerAngles(2), eulerAngles(1), eulerAngles(0)};
  }

  CaSX trans = mjpc_casadi::CASX_POSITION_ZERO;
  CaSX quat = mjpc_casadi::CASX_ORIENTATION_ZERO;  // wxyz
  void add_noise();
  constexpr int size() const { return size_byte() / sizeof(double); }
  constexpr int size_byte() const { return sizeof(trans) + sizeof(quat); }
  std::vector<double> data() const {
    std::vector<double> out(size());
    std::memcpy(out.data(), trans.ptr(), sizeof(trans));
    std::memcpy(out.data() + (sizeof(trans) / sizeof(double)), quat.ptr(), sizeof(quat));
    return out;
  }

  void from_data(const double* data) {
    std::memcpy(trans.ptr(), data, sizeof(trans));
    std::memcpy(quat.ptr(), data + (sizeof(trans) / sizeof(double)), sizeof(quat));
  }
};

struct CIOVelocity {
  CaSX linear_vel = mjpc_casadi::CASX_3D_ZERO;
  CaSX angular_vel = mjpc_casadi::CASX_3D_ZERO;
  void add_noise();
  constexpr int size() const { return size_byte() / sizeof(double); }
  constexpr int size_byte() const { return sizeof(linear_vel) + sizeof(angular_vel); }
  std::vector<double> data() const {
    std::vector<double> out(size());
    std::memcpy(out.data(), linear_vel.ptr(), sizeof(linear_vel));
    std::memcpy(out.data() + (sizeof(linear_vel) / sizeof(double)), angular_vel.ptr(), sizeof(angular_vel));
    return out;
  }
  void from_data(const double* data) {
    std::memcpy(linear_vel.ptr(), data, sizeof(linear_vel));
    std::memcpy(angular_vel.ptr(), data + (sizeof(linear_vel) / sizeof(double)), sizeof(angular_vel));
  }
};

struct CIOAcceleration {
  CaSX linear_acc = mjpc_casadi::CASX_3D_ZERO;
  CaSX angular_acc = mjpc_casadi::CASX_3D_ZERO;
  void add_noise();
  constexpr int size() const { return size_byte() / sizeof(double); }
  constexpr int size_byte() const { return sizeof(linear_acc) + sizeof(angular_acc); }
  std::vector<double> data() const {
    std::vector<double> out(size());
    std::memcpy(out.data(), linear_acc.ptr(), sizeof(linear_acc));
    std::memcpy(out.data() + (sizeof(linear_acc) / sizeof(double)), angular_acc.ptr(), sizeof(angular_acc));
    return out;
  }
  void from_data(const double* data) {
    std::memcpy(linear_acc.ptr(), data, sizeof(linear_acc));
    std::memcpy(angular_acc.ptr(), data + (sizeof(linear_acc) / sizeof(double)), sizeof(angular_acc));
  }
};

struct CIOGoal {
  CIOPose pose;
  CIOVelocity vel;
  // CIOAcceleration acc;
};

struct CIOContactMeta {
  const char* site_name = nullptr;
  const char* site_geom_name = nullptr;
  bool active = false;
};

struct CIOContact {
  int id = 0;
  // Contact force
  CaSX f = mjpc_casadi::CASX_3D_ZERO;
  // Position of applied force in the frame of the manipulated object
  CaSX ro = mjpc_casadi::CASX_3D_ZERO;
  // [0,1]: Probability of being in contact
  double c = 0;

  // Position of applied force in world frame
  CaSX r = mjpc_casadi::CASX_3D_ZERO;
  // Projection of ro onto object
  CaSX pi_O_ = mjpc_casadi::CASX_3D_ZERO;
  // Projection of applied force onto hand
  CaSX pi_H_ = mjpc_casadi::CASX_3D_ZERO;
  CaSX e_O_ = mjpc_casadi::CASX_3D_ZERO;
  CaSX e_H_ = mjpc_casadi::CASX_3D_ZERO;
  CaSX e_dot_O_ = mjpc_casadi::CASX_3D_ZERO;
  CaSX e_dot_H_ = mjpc_casadi::CASX_3D_ZERO;
  bool empty() const { return f.is_zero() && ro.is_zero() && (c == 0); }
  void add_noise();
  constexpr int size() const { return size_byte() / sizeof(double); }
  constexpr int size_byte() const { return sizeof(f) + sizeof(ro) + sizeof(c); }
  std::vector<double> data() const {
    std::vector<double> out(size());
    std::memcpy(out.data(), f.ptr(), sizeof(f));
    std::memcpy(out.data() + (sizeof(f) / sizeof(double)), ro.ptr(), sizeof(ro));
    std::memcpy(out.data() + ((sizeof(f) + sizeof(ro)) / sizeof(double)), &c, sizeof(c));
    return out;
  }
  void from_data(const double* data) {
    std::memcpy(f.ptr(), data, sizeof(f));
    std::memcpy(ro.ptr(), data + (sizeof(f) / sizeof(double)), sizeof(ro));
    std::memcpy(&c, data + ((sizeof(f) + sizeof(ro)) / sizeof(double)), sizeof(c));
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
  CIOObject(mjtObj type, int id, int geom_id, double step_size = 0.5)
      : type_(type), id_(id), geom_id_(geom_id), step_size_(step_size) {}

  void set_mj_info(const mjModel* model, const mjData* data, const mjpc::Task* task) {
    mj_model_ = model;
    mj_data_ = data;
    mj_task_ = task;
  }
  mjtObj type() const { return type_; }
  int id() const { return id_; }
  int body_id() const {
    return (mjOBJ_BODY == type_) ? id_ : (mjOBJ_SITE == type_) ? mj_model_->site_bodyid[id_] : -1;
  }
  int geom_id() const { return (mjOBJ_GEOM == type_) ? id_ : (mjOBJ_SITE == type_) ? geom_id_ : -1; }
  double mass() const { return mj_task_->QueryBodyMass(body_id()); }
  CIOPose pose() const {
    const int _body_id_ = body_id();
    return CIOPose(mjpc_casadi::from_mjpos(mj_task_->QueryBodyPos(_body_id_)),
                   mjpc_casadi::from_mjquat(mj_task_->QueryBodyQuat(_body_id_)));
  }
  CIOVelocity vel() const {
    const int _body_id_ = body_id();
    return CIOVelocity{
        .linear_vel = mjpc_casadi::from_mjvel(mj_task_->QueryBodyVel(_body_id_)),
        .angular_vel = mjpc_casadi::from_mjvel(mj_task_->QueryBodyVel(_body_id_, nullptr, false))};
  }

  CIOAcceleration acc() const {
    const int _body_id_ = body_id();
    return CIOAcceleration{
        .linear_acc = mjpc_casadi::from_mjacc(mj_task_->QueryBodyAcc(_body_id_)),
        .angular_acc = mjpc_casadi::from_mjacc(mj_task_->QueryBodyAcc(_body_id_, nullptr, false))};
  }

  CaSX get_surface_normal(const CaSX& point) const {
    const auto delta = point - pose().position();
    return delta / CaSX::norm_2(delta);
  }

  virtual CaSX project_point(const CaSX& point) const { return mjpc_casadi::CASX_POSITION_ZERO; }
  virtual void discretize() {}
  virtual bool check_inside(const CaSX& point) { return false; }

protected:
  const mjModel* mj_model_ = nullptr;
  const mjData* mj_data_ = nullptr;
  const mjpc::Task* mj_task_ = nullptr;
  mjtObj type_ = mjOBJ_UNKNOWN;
  int id_ = 0;
  int geom_id_ = 0;
  double step_size_ = 1e-3;
  double rad_bounds_ = 1e-1;
};
using CIOObjectPtr = std::shared_ptr<CIOObject>;

#if 1
// Ref: https://github.com/ctu-mrs/mrs_lib/blob/master/src/geometry/shapes.cpp
class CIOLine {
public:
  CIOLine() = default;
  CIOLine(const CaSX& p1, const CaSX& p2) : p1_(p1), p2_(p2) {}

  CaSX p1() const { return p1_; }
  CaSX p2() const { return p2_; }

  CaSX direction() const { return (p2_ - p1_); }

  CIOLine direction_cast(const CaSX& origin, const CaSX& direction) const {
    return CIOLine(origin, origin + direction);
  }

  CaSX project_point(const CaSX& p) const {
    const auto p1p = p - p1_;
    const auto p12 = p2_ - p1_;
    return p1_ + CaSX::dot(p1p, p12) / CaSX::dot(p12, p12) * p12;
  }

private:
  CaSX p1_ = mjpc_casadi::CASX_POSITION_ZERO;
  CaSX p2_ = mjpc_casadi::CASX_POSITION_ZERO;
};

class CIOTriangle {
public:
  CIOTriangle(const CaSX& p1, const CaSX& p2, const CaSX& p3) : p1_(p1), p2_(p2), p3_(p3) {}

  CaSX p1() const { return p1_; }
  CaSX p2() const { return p2_; }
  CaSX p3() const { return p3_; }

  CaSX normal() const {
    const CaSX n = CaSX::cross(p2_ - p1_, p3_ - p1_);
    return n / CaSX::norm_2(n);
  }

  CaSX center() const { return (p1_ + p2_ + p3_) / 3.0; }

  std::vector<CaSX> vertices() const {
    std::vector<CaSX> vertices;
    vertices.push_back(p1_);
    vertices.push_back(p2_);
    vertices.push_back(p3_);
    return vertices;
  }

  CaSX intersection_ray(const CIOLine& r, double epsilon) const {
    // The Möller–Trumbore algorithm
    // https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    CaSX v1 = p2_ - p1_;
    CaSX v2 = p3_ - p1_;
    CaSX h = CaSX::cross(r.direction(), v2);
    double res = (double)CaSX::dot(v1, h).scalar();
    if (res > -epsilon && res < epsilon) {
      return mjpc_casadi::CASX_POSITION_ZERO;
    }
    double f = 1.0 / res;
    CaSX s = r.p1() - p1_;
    double u = f * (double)CaSX::dot(s, h);
    if (u < 0.0 || u > 1.0) {
      return mjpc_casadi::CASX_POSITION_ZERO;
    }
    CaSX q = CaSX::cross(s, v1);
    double v = f * (double)CaSX::dot(r.direction(), q).scalar();
    if (v < 0.0 || u + v > 1.0) {
      return mjpc_casadi::CASX_POSITION_ZERO;
    }
    double t = f * (double)CaSX::dot(v2, q).scalar();
    if (t > epsilon) {
      return r.p1() + r.direction() * t;
    }
    return mjpc_casadi::CASX_POSITION_ZERO;
  }

  CaSX p1_ = mjpc_casadi::CASX_UNIT_X;
  CaSX p2_ = mjpc_casadi::CASX_UNIT_Y;
  CaSX p3_ = mjpc_casadi::CASX_UNIT_Z;
};

class CIORectangle {
public:
  CIORectangle(const CaSX& p1, const CaSX& p2, const CaSX& p3, const CaSX& p4)
      : p1_(p1), p2_(p2), p3_(p3), p4_(p4) {}
  explicit CIORectangle(const std::vector<CaSX>& points)
      : p1_(points[0]), p2_(points[1]), p3_(points[2]), p4_(points[3]) {}
  CaSX p1() const { return p1_; }
  CaSX p2() const { return p2_; }
  CaSX p3() const { return p3_; }
  CaSX p4() const { return p4_; }
  CaSX center() const { return 0.25 * (p1_ + p2_ + p3_ + p4_); }
  CaSX normal() const {
    const auto n = CaSX::cross(p2_ - p1_, p4_ - p1_);
    return n / CaSX::norm_2(n);
  }

  std::vector<CaSX> vertices() const {
    std::vector<CaSX> vertices;
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

  CaSX intersection_ray(const CIOLine& r, double epsilon) const {
    CIOTriangle t1 = triangles()[0];
    CIOTriangle t2 = triangles()[1];
    auto result = t1.intersection_ray(r, epsilon);
    if (!result.is_zero()) {
      return result;
    }
    return t2.intersection_ray(r, epsilon);
  }

  bool is_facing(const CaSX& point) const {
    CaSX towards_point = point - center();
    double dot_product = (double)CaSX::dot(towards_point, normal()).scalar();
    return dot_product > 0;
  }

  CaSX project_point(const CaSX& point) const {
    const auto rect_normal = normal();
    return point - CaSX::dot(rect_normal, point) * rect_normal;
  }

private:
  CaSX p1_ = mjpc_casadi::CASX_POSITION_ZERO;
  CaSX p2_ = mjpc_casadi::CASX_UNIT_X;
  CaSX p3_ = mjpc_casadi::CASX_UNIT_Y;
  CaSX p4_ = mjpc_casadi::CASX_UNIT_Z;
};

class CIOEllipse {
public:
  CIOEllipse() = default;
  CIOEllipse(const CaSX& center, const CaSX& orientation, double a, double b)
      : center_point_(center), absolute_orientation_(orientation), major_semi_(a), minor_semi_(b) {}

  double p1() const { return major_semi_; }
  double p2() const { return minor_semi_; }

  const CaSX center() const { return center_point_; }
  const CaSX orientation() const { return absolute_orientation_; }

private:
  double major_semi_ = 0;
  double minor_semi_ = 0;
  CaSX center_point_ = mjpc_casadi::CASX_POSITION_ZERO;
  CaSX absolute_orientation_ = mjpc_casadi::CASX_ORIENTATION_ZERO;
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
  CIOCuboid(mjtObj type, int id, int geom_id) : CIOObject(type, id, geom_id) {}
  CIOCuboid(mjtObj type, int id, int geom_id, const CaSX& center, const CaSX& radius, const CaSX& orientation)
      : CIOObject(type, id, geom_id) {
    // set_points(center, radius, orientation);
  }

  void set_points(const CaSX& center, const CaSX& radius, const CaSX& orientation) {
    const auto x = (double)radius(0).scalar();
    const auto y = (double)radius(1).scalar();
    const auto z = (double)radius(2).scalar();

    const Eigen::Quaterniond quat = mjpc_casadi::to_eigen_quat(orientation);
    CaSX p0 = CaSX{x, -y, z};
    auto p0_quat = quat * Eigen::Vector3d{x, -y, z};
    CaSX p0_ori = CaSX{p0_quat[0], p0_quat[1], p0_quat[2]};

    CaSX p1 = CaSX{x, y, -z};
    auto p1_quat = quat * Eigen::Vector3d{x, y, -z};
    CaSX p1_ori = CaSX{p1_quat[0], p1_quat[1], p1_quat[2]};

    CaSX p2 = CaSX{x, y, z};
    auto p2_quat = quat * Eigen::Vector3d{x, y, z};
    CaSX p2_ori = CaSX{p2_quat[0], p2_quat[1], p2_quat[2]};

    CaSX p3 = CaSX{x, -y, z};
    auto p3_quat = quat * Eigen::Vector3d{x, -y, z};
    CaSX p3_ori = CaSX{p3_quat[0], p3_quat[1], p3_quat[2]};

    CaSX p4 = CaSX{-x, y, z};
    auto p4_quat = quat * Eigen::Vector3d{-x, y, z};
    CaSX p4_ori = CaSX{p4_quat[0], p4_quat[1], p4_quat[2]};

    CaSX p5 = CaSX{-x, -y, -z};
    auto p5_quat = quat * Eigen::Vector3d{-x, -y, -z};
    CaSX p5_ori = CaSX{p5_quat[0], p5_quat[1], p5_quat[2]};

    CaSX p6 = CaSX{-x, -y, z};
    auto p6_quat = quat * Eigen::Vector3d{-x, -y, z};
    CaSX p6_ori = CaSX{p6_quat[0], p6_quat[1], p6_quat[2]};

    CaSX p7 = CaSX{-x, y, z};
    auto p7_quat = quat * Eigen::Vector3d{-x, y, z};
    CaSX p7_ori = CaSX{p7_quat[0], p7_quat[1], p7_quat[2]};

    p0 = center + p0_ori;
    p1 = center + p1_ori;
    p2 = center + p2_ori;
    p3 = center + p3_ori;

    p4 = center + p4_ori;
    p5 = center + p5_ori;
    p6 = center + p6_ori;
    p7 = center + p7_ori;

    points_.clear();
    points_.push_back(p0);
    points_.push_back(p1);
    points_.push_back(p2);
    points_.push_back(p3);
    points_.push_back(p4);
    points_.push_back(p5);
    points_.push_back(p6);
    points_.push_back(p7);
  }

  std::vector<CaSX> lookup_points(int face_idx) const {
    std::vector<CaSX> lookup;
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

  std::vector<CaSX> vertices() const { return points_; }

  CIORectangle get_rectangle(int face_idx) const { return CIORectangle{lookup_points(face_idx)}; }

  CaSX center() const {
    CaSX point_sum = points_[0];
    for (int i = 1; i < 8; i++) {
      point_sum += points_[i];
    }
    return point_sum / 8.0;
  }

  std::vector<CaSX> intersection_ray(const CIOLine& r, double epsilon) const {
    std::vector<CaSX> ret;
    for (int i = 0; i < 6; i++) {
      CIORectangle side = get_rectangle(i);
      auto side_intersect = side.intersection_ray(r, epsilon);
      if (!side_intersect.is_zero()) {
        ret.push_back(side_intersect);
      }
    }
    return ret;
  }

  CaSX project_point(const CaSX& point) const override {
    static constexpr float k = 1.e4;
    static constexpr int num_faces = 6;

    // Initialize p_nearest as a zero matrix
    CaSX p_nearest = CaSX::zeros(num_faces, 3);
    for (auto j = 0; j < num_faces; ++j) {
      p_nearest(j) = get_rectangle(j).project_point(point);
    }

    // Transpose the point and tile it
    CaSX p_mat = CaSX::zeros(num_faces, 3);
    for (auto j = 0; j < num_faces; ++j) {
      p_mat(j) = point;
    }

    // Create ones vector
    const auto ones_vec = CaSX::ones(num_faces);

    // Calculate nu, using a softmin instead of a hardmin to make function smooth
    // https://research.cs.wisc.edu/zhu/space2/TTP2/advanced_image_selection/bin/toolbox/doc/classify/softmin.html
    // Let d be a vector.  Then the softmin of d is defined as:
    // s = exp(-d / sigma ^ 2) / sum(exp(-d / sigma ^ 2))
    // The softmin is a way of taking a dissimilarity(distance) vector d and
    // converting it to a similarity vector s,
    // such that sum(s) == 1.
    const auto d = CaSX::norm_1(CaSX::sq(p_mat - p_nearest));
#if 1
    const auto sigma = k;
    CaSX nu = CaSX::exp(-CaSX::vec(d) / (sigma * sigma));
#else
    CaSX nu = CaSX::vec(ones_vec) / (CaSX::vec(ones_vec) + CaSX::vec(d) * k);
#endif

    // Normalize nu
    nu /= CaSX::norm_2(nu);

    // Tile nu for broadcasting
    // CaSX nu_tiled = nu.replicate(1, 3);
    CaSX nu_tiled = CaSX::repmat(nu, 1, 3);

    // Calculate closest point
    const auto closest_point = CaSX::sum2(CaSX::vec(nu_tiled) * CaSX::vec(p_nearest));
    return closest_point;
  }

private:
  std::vector<CaSX> points_ = std::vector<CaSX>(8, mjpc_casadi::CASX_POSITION_ZERO);
};

class CIOCylinder : public CIOObject {
public:
  enum {
    BOTTOM = 0,
    TOP = 1,
  };
  CIOCylinder(const CaSX& center, double radius, double height, const CaSX& orientation)
      : center_point_(center), radius_(radius), height_(height), absolute_orientation_(orientation) {}

  CaSX center() const { return center_point_; }
  CaSX orientation() const { return absolute_orientation_; }

  double r() const { return radius_; }
  double h() const { return height_; }

  CIOEllipse get_cap(int index) const {
    CIOEllipse e;
    CaSX ellipse_center;
    switch (index) {
      case BOTTOM:
        ellipse_center = center() - orientation() * (0.5 * h() * mjpc_casadi::CASX_UNIT_Z);
        e = CIOEllipse(ellipse_center, orientation(), r(), r());
        break;
      case TOP:
        ellipse_center = center() + orientation() * (0.5 * h() * mjpc_casadi::CASX_UNIT_Z);
        e = CIOEllipse(ellipse_center, orientation(), r(), r());
        break;
    }
    return e;
  }

private:
  CaSX center_point_ = mjpc_casadi::CASX_POSITION_ZERO;
  double radius_ = 0;
  double height_ = 0;
  CaSX absolute_orientation_ = mjpc_casadi::CASX_ORIENTATION_ZERO;
};

class CIOCone : public CIOObject {
public:
  CIOCone(const CaSX& origin_point, double angle, double height, const CaSX& absolute_direction)
      : origin_point_(origin_point),
        angle_(angle),
        height_(height),
        absolute_direction_(CaSX::norm_2(absolute_direction)) {}
  CaSX origin() const { return origin_point_; }
  CaSX direction() const { return absolute_direction_; }
  CaSX center() const { return origin() + (0.5 * h()) * direction(); }

  double theta() const { return angle_; }
  double h() const { return height_; }

  CIOEllipse get_cap() const {
    const CaSX ellipse_center = origin() + direction() * h();
    const auto ellipse_orientation = Eigen::Quaterniond::FromTwoVectors(
        Eigen::Vector3d::UnitZ(), mjpc_casadi::to_eigen_vector(direction(), 3));
    double cap_radius = std::tan(theta()) * h();
    CIOEllipse e(ellipse_center, mjpc_casadi::from_eigen_quat(ellipse_orientation), cap_radius, cap_radius);
    return e;
  }

  CaSX project_point(const CaSX& point) const override {
    const CaSX point_vec = point - origin();
    double point_axis_angle = (double)CaSX::acos(CaSX::dot(point_vec, direction()) /
                                                 (CaSX::norm_2(point_vec) * CaSX::norm_2(direction())))
                                  .scalar();

    /* CaSX axis_projection = this->cone_axis_projector * point_vec + origin(); */

    CaSX rot_axis = CaSX::cross(direction(), point_vec);
    rot_axis = rot_axis / CaSX::norm_2(rot_axis);

    Eigen::AngleAxis<double> quat(this->angle_ - point_axis_angle, mjpc_casadi::to_eigen_vector3(rot_axis));

    const CaSX point_on_cone = mjpc_casadi::from_eigen_vector3(Eigen::Vector3d(
        quat * mjpc_casadi::to_eigen_vector3(point_vec) + mjpc_casadi::to_eigen_vector3(origin())));

    CaSX vec_point_on_cone = point_on_cone - origin();
    vec_point_on_cone = vec_point_on_cone / CaSX::norm_2(vec_point_on_cone);

    double beta = this->angle_ - point_axis_angle;

    if (point_axis_angle < this->angle_) {
      return origin() + vec_point_on_cone * CaSX::cos(beta) * CaSX::norm_2(point_vec);
    } else if ((point_axis_angle >= this->angle_) &&
               (point_axis_angle - this->angle_) <= M_PI_2) {  // TODO: is this condition correct?
      return origin() +
             vec_point_on_cone * CaSX::cos(point_axis_angle - this->angle_) * CaSX::norm_2(point_vec);
    } else {
      return mjpc_casadi::CASX_POSITION_ZERO;
    }
  }

private:
  CaSX origin_point_ = mjpc_casadi::CASX_POSITION_ZERO;
  double angle_ = 0;
  double height_ = 0;
  CaSX absolute_direction_ = mjpc_casadi::CASX_ORIENTATION_ZERO;
};

class CIOSphere : public CIOObject {
public:
  CIOSphere() = default;
  CIOSphere(mjtObj type, int id, int geom_id, double step_size = 0.5)
      : CIOObject(type, id, geom_id, step_size), radius_(mj_model_ ? mj_model_->geom_size[geom_id] : 0) {}

  // Projects the given point onto the surface of this object
  CaSX project_point(const CaSX& point) const override {
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
