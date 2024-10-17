#pragma once

#include <Eigen/Core>
#include <chrono>
#include <memory>
#include <random>
#include <utility>

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
  void set_dynamics(const CIOPose& pose, const CIOVelocity& vel, const CIOAcceleration& acc) {
    pose_ = pose;
    vel_ = vel;
    acc_ = acc;
  }

  bool check_collisions(const CIOObject& other_object) { return false; }

  Eigen::Vector3d get_surface_normal(const Eigen::Vector3d& point) {
    return (point - pose_.position()).normalized();
  }

  virtual Eigen::Vector3d project_point(const Eigen::Vector3d& point) = 0;

protected:
  int id_ = 0;
  CIOPose pose_;
  CIOVelocity vel_;
  CIOAcceleration acc_;
  double step_size_ = 0.001;
  double rad_bounds_ = 1e-1;
};
using CIOObjectPtr = std::shared_ptr<CIOObject>;

class CIOSphere : public CIOObject {
public:
  CIOSphere() = default;
  CIOSphere(int id, double radius, const Eigen::Vector3d& pos, const Eigen::Vector3d& vel,
            double step_size = 0.5)
      : CIOObject(id, CIOPose(pos), CIOVelocity{.linear_vel = vel}, step_size), radius_(radius) {}
  double radius_ = 0.;

  // Projects the given point onto the surface of this object
  Eigen::Vector3d project_point(const Eigen::Vector3d& point) override {
    return pose_.position() + (radius_ * get_surface_normal(point));
  }
};
using CIOSpherePtr = std::shared_ptr<CIOSphere>;

using CIOFinger = CIOSphere;
using CIOFingerPtr = std::shared_ptr<CIOFinger>;
using CIOContactMap = std::map<CIOObjectPtr, CIOContact>;

class CIOWorld {
public:
  using CIOWorldPtr = std::shared_ptr<CIOWorld>;
  CIOWorld(CIOObjectPtr manip_obj, std::vector<CIOFingerPtr> fingers, const CIOContactMap& contact_states_,
           std::function<void()> traj_func)
      : manip_obj_(std::move(manip_obj)),
        fingers_(std::move(fingers)),
        contact_states_(contact_states_),
        traj_func_(std::move(traj_func)) {}

  std::map<int /*obj_id*/, Eigen::Vector3d> pi_O_;
  std::map<int /*obj_id*/, Eigen::Vector3d> pi_H_;
  std::map<int /*obj_id*/, Eigen::Vector3d> e_O_;
  std::map<int /*obj_id*/, Eigen::Vector3d> e_H_;
  std::map<int /*obj_id*/, Eigen::Vector3d> e_dot_O_;
  std::map<int /*obj_id*/, Eigen::Vector3d> e_dot_H_;

  void set_dynamics(int obj_idx, const CIOPose& pose, const CIOVelocity& vel, const CIOAcceleration& acc) {
    auto objects = get_all_objects();
    objects[obj_idx]->set_dynamics(pose, vel, acc);
  }

  void set_contact_state(int obj_id, const Eigen::Vector3d& f, const Eigen::Vector3d& ro, double c) {
    for (auto& [object, contact] : contact_states_) {
      if (object->id() == obj_id) {
        contact = CIOContact{.f = f, .ro = ro, .c = c};
      }
    }
  }

  void set_e_vars(const CIOConfig& config, const CIOWorldPtr& world_tm1 = nullptr) {
    const auto obj_pose = manip_obj_->pose();
    for (auto& [object, contact] : contact_states_) {
      const auto r = obj_pose.position() + contact.ro;
      const auto ci = object->id();
      pi_H_[ci] = object->project_point(r);
      pi_O_[ci] = manip_obj_->project_point(r);
      e_H_[ci] = pi_H_[ci] - r;
      e_O_[ci] = pi_O_[ci] - r;

      if (world_tm1) {
        e_dot_H_[ci] = CIOUtils::calc_derivative(e_H_[ci], world_tm1->e_H_[ci], config.delT);
        e_dot_O_[ci] = CIOUtils::calc_derivative(e_O_[ci], world_tm1->e_O_[ci], config.delT);
      } else {
        e_dot_H_[ci] = Eigen::Vector3d::Zero();
        e_dot_O_[ci] = Eigen::Vector3d::Zero();
      }
    }
  }

  std::vector<CIOObjectPtr> get_all_objects() const {
    std::vector<CIOObjectPtr> objects = {manip_obj_};
    for (const auto& finger : fingers_) {
      objects.push_back(finger);
    }
    return objects;
  }

  std::vector<CIOObservation> get_observation() const {
    std::vector<CIOObservation> s;
    for (const auto& obj : get_all_objects()) {
      s.push_back(CIOObservation{.obj_id = obj->id(), .pose = obj->pose(), .vel = obj->vel()});
    }
    for (const auto& [obj, contact] : contact_states_) {
      s.push_back(
          CIOObservation{.obj_id = obj->id(), .pose = obj->pose(), .vel = obj->vel(), .contact = contact});
    }
    return s;
  }

  // Ref: https://gist.github.com/lorenzoriano/5414671
  template <typename T>
  static std::vector<T> linspace(double start, double end, int num) {
    std::vector<T> linspaced;

    if (0 != num) {
      if (1 == num) {
        linspaced.push_back(static_cast<T>(start));
      } else {
        double delta = (end - start) / (num - 1);

        for (auto i = 0; i < (num - 1); ++i) {
          linspaced.push_back(static_cast<T>(start + delta * i));
        }
        // ensure that start and end are exactly the same as the input
        linspaced.push_back(static_cast<T>(end));
      }
    }
    return linspaced;
  }

  static std::vector<std::vector<double>> linspace_vectors(const std::vector<double>& vec0,
                                                           const std::vector<double>& vec1, int num_steps) {
    const auto l = vec0.size();
    std::vector<std::vector<double>> out_vec(l);
    for (auto j = 0; j < l; ++j) {
      const auto left = vec0[j];
      const auto right = vec1[j];
      out_vec[j] = linspace<double>(left, right, num_steps);
    }
    return out_vec;
  }

  static std::vector<double> add_gaussian_noise(const std::vector<double>& vec) {
    // perturb all vars by gaussian noise
    const double mean = 0.;
    const double var = 0.01;
    std::vector<double> out_vec = vec;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(mean, var);

    for (auto j = 0; j < vec.size(); ++j) {
      out_vec[j] += std::abs(distribution(generator));
    }
    return out_vec;
  }

  static void calc_obj_dynamics(const std::vector<CIOObservation>& observations, const CIOWorldPtr& world,
                                const CIOConfig& config) {
    double time;
    std::vector<double> times, parameters;
    int nu, num_spline_points;

#if 0
    for (auto& obs : observations) {
      const auto pos = obs.pose.position();
      const auto lin_vel = obs.vel.linear_vel;
      std::vector<double> pos_ = {pos[0], pos[1], pos[2], lin_vel[0], lin_vel[1], lin_vel[2]};
      mjpc::CubicInterpolation(pos_.data(), time, times, parameters.data(), nu, num_spline_points);
    }
#endif

    std::vector<Eigen::Vector3d> pose_traj_K;
    std::vector<Eigen::Vector3d> vel_traj_K;
    for (auto& obs : observations) {
      pose_traj_K.push_back(obs.pose.position());
      vel_traj_K.push_back(obs.vel.linear_vel);
    }

    std::vector<std::pair<std::vector<double>, std::vector<double>>> splines;
    for (auto i = 0; i < 3; ++i) {
      auto x = linspace<double>(0., config.T_final(), config.K + 1);
      auto y_pos = std::vector<double>(config.K + 1);
      auto y_vel = std::vector<double>(config.K + 1);
      for (auto k = 0; k < config.K + 1; ++k) {
        y_pos[k] = pose_traj_K[k][i];
        y_vel[k] = vel_traj_K[k][i];
      }
      std::pair<std::vector<double> /*pos*/, std::vector<double> /*vel*/> spline;
      mjpc::CubicInterpolation(spline.first.data(), time, x, y_pos.data(), 3, x.size());
      mjpc::CubicInterpolation(spline.second.data(), time, x, y_vel.data(), 3, x.size());
      splines.push_back(std::move(spline));
    }

    std::vector<Eigen::Vector3d> pose_traj_T;
    int k = 0;
    times = linspace<double>(0., config.T_final(), config.T_steps() + 1);
    for (const auto t : times) {
      if (fmod(t, config.delT_phase) != 0.) {
        pose_traj_T.push_back(pose_traj_K[k]);
        k++;
      } else {
        // TODO
        for (auto i = 0; i < 3; ++i) {
          pose_traj_T.push_back(
              Eigen::Vector3d(splines[i].first[0], splines[i].first[1], splines[i].first[2]));
        }
      }
    }

    std::vector<Eigen::Vector3d> vel_traj_T;
    for (auto t = 1; t < config.T_steps() + 1; ++t) {
      vel_traj_T.push_back(CIOUtils::calc_derivative(pose_traj_T[t], pose_traj_T[t - 1], config.delT));
    }

    std::vector<Eigen::Vector3d> acc_traj_T;
    for (auto t = 1; t < config.T_steps() + 1; ++t) {
      vel_traj_T.push_back(CIOUtils::calc_derivative(vel_traj_T[t], vel_traj_T[t - 1], config.delT));
    }

    // return std::make_tuple(pose_traj_T, vel_traj_T, acc_traj_T);
  }

  CIOObjectPtr manip_obj_ = nullptr;
  std::vector<CIOFingerPtr> fingers_;
  CIOContactMap contact_states_;
  std::function<void()> traj_func_;
};
using CIOWorldPtr = CIOWorld::CIOWorldPtr;
