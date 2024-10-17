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
#include <mujoco/mujoco.h>

#include "mjpc/planners/cio/cio_common.h"
#include "mjpc/planners/cio/cio_util.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

// NOTE: CONSIDER USING [mj_geomDistance] between finger tip sites & object sites
class CIOTrajectory : public mjpc::Trajectory {
public:
  using CIOTrajectoryPtr = std::shared_ptr<CIOTrajectory>;
  CIOTrajectory() = default;

  CIOConfig GetConfig() const { return config_; }
  void SetTimestep(double timestep) { config_.delT = timestep; }

  void SetContactState(CIOObjectPtr obj, std::vector<CIOContact> contact_list) {
    contact_states_.emplace(std::move(obj), std::move(contact_list));
  }

  bool HasEnvData() const { return manip_obj_ && !fingers_.empty() && !contact_states_.empty(); }
  void Setup(const mjModel* model, const mjData* data, const mjpc::Task* task) {
    // [Model, data]
    mj_model_ = model;
    mj_data_ = data;

    // [Manip obj]
    const auto manip_obj_body_id = task->QueryBodyId("object");
    const auto manip_obj_geom_id = task->QueryGeomId("object");
    const auto* obj_pos = task->QueryBodyPos(manip_obj_body_id);
    const auto* obj_rot = task->QueryBodyQuat(manip_obj_body_id);
    const auto obj_size = task->QueryGeomSize("object");
    const auto obj_center = Eigen::Vector3d(obj_pos[0], obj_pos[1], obj_pos[2]);
    const auto obj_radius = Eigen::Vector3d(obj_size.data());
    if (manip_obj_) {
      std::dynamic_pointer_cast<CIOCuboid>(manip_obj_)
          ->set_points(obj_center, obj_radius, Eigen::Quaterniond(obj_rot));
    } else {
      manip_obj_ = std::make_shared<CIOCuboid>(manip_obj_body_id, manip_obj_geom_id, obj_center, obj_radius,
                                               Eigen::Quaterniond(obj_rot));
    }
    manip_obj_->set_mj_info(model, data, task);

    // [Fingers]
    const auto finger0_body_id = task->QueryBodyId("finger_a");
    const auto finger1_body_id = task->QueryBodyId("finger_b");
    if (fingers_.empty()) {
      const auto fingertip_radius = task->QueryGeomSizeMax("finger_a");
      auto finger0 =
          std::make_shared<CIOSphere>(finger0_body_id, task->QueryGeomId("finger_a"), fingertip_radius);
      finger0->set_mj_info(model, data, task);
      auto finger1 =
          std::make_shared<CIOSphere>(finger1_body_id, task->QueryGeomId("finger_b"), fingertip_radius);
      finger1->set_mj_info(model, data, task);
      fingers_ = std::vector{std::move(finger0), std::move(finger1)};
    }

    // [Fingers' Contacts]
    int ncon = data->ncon;
    CIOContactMap contact_states;
    std::vector<CIOContact> finger0_contacts;
    std::vector<CIOContact> finger1_contacts;
    for (int i = 0; i < ncon; i++) {
      auto contact_i = data->contact[i];
      int body1 = model->geom_bodyid[contact_i.geom1];
      int body2 = model->geom_bodyid[contact_i.geom2];
      // nothing to do for excluded contacts
      if (contact_i.efc_address < 0) {
        continue;
      }

      // Get contact force/torque, rotate into traj frame, then site frame.
      // Note that contact.frame is column major.
      mjtNum conforce[6], conray[3];
      // get contact force:torque in contact frame
      mj_contactForce(model, data, i, conforce);

      // convert contact normal force to global frame, normalize
      mju_mulMatTVec3(conray, contact_i.frame, conforce);
      mju_normalize3(conray);

      auto cio_contact = CIOContact{.f = Eigen::Vector3d(conray),
                                    .ro = Eigen::Vector3d(contact_i.pos),
                                    .c = double((contact_i.dist > 0) ? 1 : 0)};
      if ((body1 == finger0_body_id) || (body2 == finger0_body_id)) {
        finger0_contacts.emplace_back(std::move(cio_contact));
      } else if ((body1 == finger1_body_id) || (body2 == finger1_body_id)) {
        finger1_contacts.emplace_back(std::move(cio_contact));
      }
    }

    contact_states.emplace(fingers_[0], std::move(finger0_contacts));
    contact_states.emplace(fingers_[1], std::move(finger1_contacts));
    contact_states_ = std::move(contact_states);

    // Configure traj
    SetTimestep(model->opt.timestep);
    SetEVars();
  }

  void SetEVars() {
    const auto manip_obj_pose = manip_obj_->pose();
    for (auto& [contact_obj, contact_list] : contact_states_) {
      for (auto i = 0; i < contact_list.size(); ++i) {
        auto& contact_i = contact_list[i];
        const auto& r = contact_i.r = manip_obj_pose.position() + contact_i.ro;
        contact_i.pi_H_ = contact_obj->project_point(r);
        contact_i.pi_O_ = manip_obj_->project_point(r);
        contact_i.e_H_ = contact_i.pi_H_ - r;
        contact_i.e_O_ = contact_i.pi_O_ - r;

        auto traj_contact_i = (prev_traj_contact_states_.contains(contact_obj) &&
                               (i < prev_traj_contact_states_.at(contact_obj).size()))
                                  ? prev_traj_contact_states_.at(contact_obj)[i]
                                  : contact_i;
        if (traj_contact_i.empty()) {
          traj_contact_i = contact_i;
        }
        contact_i.e_dot_H_ = CIOUtils::calc_derivative(contact_i.e_H_, traj_contact_i.e_H_, config_.delT);
        contact_i.e_dot_O_ = CIOUtils::calc_derivative(contact_i.e_H_, traj_contact_i.e_H_, config_.delT);
      }
    }
    prev_traj_contact_states_ = contact_states_;
  }

  // Contact-invariant cost
  double L_Contacts() const {
    double cost = 0;
    for (const auto& [_, contact_list] : contact_states_) {
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
  double L_Kinematics() const {
#if 1
    return GetFingersSelfCollisionNum();
#else
    mjtNum fromto_new[6] = {0};
    mjtNum margin = 0.01;
    return mj_geomDistance(mj_model_, mj_data_, fingers_[0]->geom_id(), fingers_[1]->geom_id(), margin,
                           fromto_new);
#endif
  }

  // Physics-violation cost
  double L_Physics() const {
    double newton_cost = 0;

    // 1- Total (linear) forces on object
    Eigen::Vector3d f_total = Eigen::Vector3d::Zero();
    for (const auto& [contact_obj, contact_list] : contact_states_) {
      for (const auto& contact : contact_list) {
        f_total += contact.c * contact.f;
      }
    }
#if CIO_USE_EXT_OBJ_WRENCH
    const auto f_ext = mj_data_->cfrc_ext ? Eigen::Vector3d(mj_data_->cfrc_ext[6 * manip_obj_->id() + 3],
                                                            mj_data_->cfrc_ext[6 * manip_obj_->id() + 4],
                                                            mj_data_->cfrc_ext[6 * manip_obj_->id() + 5])
                                          : Eigen::Vector3d::Zero();
    f_total += f_ext;
#endif
    // Cost of change in linear momentum over time
    // https://courses.lumenlearning.com/suny-physics/chapter/8-1-linear-momentum-and-force
    const auto p_dot = manip_obj_->mass() * (manip_obj_->acc().linear_acc);
    newton_cost = (f_total - p_dot).squaredNorm();

    // 2- Total (angular) torques on object
    Eigen::Vector3d m_total = Eigen::Vector3d::Zero();
    for (const auto& [contact_obj, contact_list] : contact_states_) {
      for (const auto& contact : contact_list) {
        m_total += (contact.c * contact.f).cross(contact.ro - contact_obj->pose().position());
      }
    }
#if CIO_USE_EXT_OBJ_WRENCH
    const auto m_ext = mj_data_->cfrc_ext ? Eigen::Vector3d(mj_data_->cfrc_ext[6 * manip_obj_->id()],
                                                            mj_data_->cfrc_ext[6 * manip_obj_->id() + 1],
                                                            mj_data_->cfrc_ext[6 * manip_obj_->id() + 2])
                                          : Eigen::Vector3d::Zero();
    m_total += m_ext;
#endif
    //  Cost of change in angular momentum over time
    //  https://courses.lumenlearning.com/suny-physics/chapter/10-5-angular-momentum-and-its-conservation
    //  std::vector<mjtNum> angular_mat(3 * mj_model_->nv);
    //  mj_angmomMat(mj_model_, mj_data_, angular_mat.data(), manip_obj_->id());
    auto* l = &mj_data_->subtree_angmom[3 * manip_obj_->id()];
    const auto l_dot = Eigen::Vector3d(l[0], l[1], l[2]) / mj_model_->opt.timestep;
    newton_cost += (m_total - l_dot).squaredNorm();

    // 3- Force regularization cost
    double force_reg_cost = 0.0;
    for (const auto& [contact_obj, contact_list] : contact_states_) {
      for (const auto& contact : contact_list) {
        force_reg_cost += contact.f.squaredNorm();
      }
    }
    force_reg_cost *= config_.lamb;

    // 4- [L_cone]: Constrain contact force to lie in the friction cone of the contact surface
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

  double L_Task(const std::vector<CIOGoal>& goals) {
    // NOTE:
    // 1. This task cost should only be accounted frame-wise, meaning each frame of the horizon has it
    // differently. This is already made sure in [Trajectory::UpdateReturn()].
    // 2. In MJPC, Task::CostValue() which invoke Residual() should already have cost specified through
    // residual (BaseResidualFn::CostTerms()), so this function is only kept for reference!

    double task_cost = 0.0;
    for (const auto& goal : goals) {
      // Position
      const auto obj_pos = manip_obj_->pose().position();
      const auto goal_pos = goal.pose.position();
      task_cost += (obj_pos - goal_pos).squaredNorm();

      // Velocity
      const auto obj_vel = manip_obj_->vel().linear_vel;
      const auto goal_vel = goal.vel.linear_vel;
      task_cost += (obj_vel - goal_vel).squaredNorm();
    }

    // Small acceleration constraint
    double accel_cost = 0.0;
    for (const auto& [obj_id, obj] : GetAllObjects()) {
      accel_cost += pow(obj->acc().linear_acc.norm(), 2);
    }
    accel_cost *= config_.lamb;

    return accel_cost + task_cost;
  }

  void CalculateNonResidualCost(const mjModel* model, mjData* data, const mjpc::Task* task,
                                int t /*horizon frame*/) override {
    // Setu
    Setup(model, data, task);

    // Calculate cost
    non_residual_costs[t] = TotalCost();
  }

  double TotalCost(int stage_idx = 0) const {
    if (!HasEnvData()) {
      return 0;
    }
    double ci = 0.0, phys = 0.0, kinem = 0.0, task = 0.0;

    const auto& stage_weight = config_.stage_weights[stage_idx];
    ci += stage_weight.w_CI * L_Contacts();
    if (ci > 0) {
      phys += stage_weight.w_physics * L_Physics();
    }
    kinem += stage_weight.w_kinematics * L_Kinematics();
#if 0
    // Already accounted for in [BaseResidualFn::CostTerms()]
    task += stage_weight.w_task * L_Task(goals);
#endif
    return ci + phys + kinem + task;
  }

  std::map<int, CIOObjectPtr> GetAllObjects() const {
    auto objects = manip_obj_ ? std::map<int, CIOObjectPtr>{{manip_obj_->id(), manip_obj_}}
                              : std::map<int, CIOObjectPtr>{};
    for (const auto& finger : fingers_) {
      objects.emplace(finger->id(), finger);
    }
    return objects;
  }

  std::vector<CIOObservation> GetObservations() const {
    std::vector<CIOObservation> s;
    for (const auto& [obj_id, obj] : GetAllObjects()) {
      s.push_back(CIOObservation{
          .obj_id = obj_id,
          .pose = obj->pose(),
          .vel = obj->vel(),
          .contacts = contact_states_.contains(obj) ? contact_states_.at(obj) : std::vector<CIOContact>{}});
    }
    return s;
  }

  std::vector<double> GetObservationsData(bool with_noise = true) const {
    std::vector<double> data;
    auto observations = GetObservations();
    for (auto& obs : observations) {
      if (with_noise) {
        obs.add_noise();
      }
      const auto obs_i_data = obs.data();
      std::copy(obs_i_data.begin(), obs_i_data.end(), std::back_inserter(data));
    }
    return data;
  }

  std::vector<CIOObservation> CalcObjDynamics() const {
    const std::vector<CIOObservation> observations = GetObservations();

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

  CIOContactMap GetContactStates() const { return contact_states_; }
  std::vector<CIOContact> GetObjectContacts(const CIOObjectPtr& object) const {
    return contact_states_.contains(object) ? contact_states_.at(object) : std::vector<CIOContact>{};
  }

  CIOContact get_contact(const CIOObjectPtr& object, int contact_idx) const {
    auto contacts = GetObjectContacts(object);
    return contacts.empty() ? CIOContact() : contacts[contact_idx];
  }

  std::vector<CIOContact> GetSmoothContacts(const CIOObjectPtr& object) const {
    auto contacts = GetObjectContacts(object);
    if (contacts.empty()) {
      return {};
    }

    // CIOTODO
    auto s0 = GetObservations();
    std::vector<CIOContact> contact_traj_K;
    for (auto k = 0; k < config_.K; ++k) {
      for (const auto& s0_i : s0) {
        for (const auto& s0_i_contact : s0_i.contacts) {
          contact_traj_K.push_back(s0_i_contact);
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

  std::vector<CIOTrajectoryPtr> DynamicTraj() const {
    // Dynamics
    std::vector<std::vector<CIOObservation>> all_dyn_info;
    for (const auto& [obj_id, obj] : GetAllObjects()) {
      all_dyn_info.push_back(CalcObjDynamics());
    }

    // Contacts
    CIOContactMap all_contact_info;
    for (const auto& [contact_obj, contact_list] : contact_states_) {
      all_contact_info.insert_or_assign(contact_obj, GetSmoothContacts(contact_obj));
    }

    // Fill into list of new trajs
    std::vector<CIOTrajectoryPtr> trajs;
    for (int t = 0; t < config_.T_steps() + 1; ++t) {
      CIOTrajectoryPtr traj_t = std::make_shared<CIOTrajectory>(*this);

      // World's objs contact
      for (const auto& [contact_obj, contact_list] : contact_states_) {
        auto obj_contact_list = all_contact_info.at(contact_obj);
        if (!obj_contact_list.empty()) {
          traj_t->SetContactState(contact_obj, std::move(obj_contact_list));
        }
      }

      traj_t->SetEVars();
      trajs.emplace_back(std::move(traj_t));
    }
    return trajs;
  }

  bool CheckCollisionj(const CIOObjectPtr& obj0, const CIOObjectPtr& obj1) {
    for (auto i = 0; i < mj_data_->ncon; ++i) {
      const auto& contact = mj_data_->contact[i];
      const auto obj0_geomid = obj0->geom_id();
      const auto obj1_geomid = obj1->geom_id();
      if (((contact.geom[0] == obj0_geomid) && (contact.geom[1] == obj1_geomid)) ||
          ((contact.geom[0] == obj1_geomid) && (contact.geom[1] == obj0_geomid))) {
        return true;
      }
    }
    return false;
  }

  int GetFingersSelfCollisionNum() const {
    int contact_count = 0;
    for (auto i = 0; i < mj_data_->ncon; ++i) {
      const auto& contact = mj_data_->contact[i];
      const auto finger0_geomid = fingers_[0]->geom_id();
      const auto finger1_geomid = fingers_[1]->geom_id();
      if (((contact.geom[0] == finger0_geomid) && (contact.geom[1] == finger1_geomid)) ||
          ((contact.geom[0] == finger1_geomid) && (contact.geom[1] == finger0_geomid))) {
        contact_count++;
      }
    }
    return contact_count;
  }

private:
  const mjModel* mj_model_ = nullptr;
  const mjData* mj_data_ = nullptr;
  CIOObjectPtr manip_obj_ = nullptr;
  std::vector<CIOFingerPtr> fingers_;
  CIOContactMap contact_states_;
  CIOContactMap prev_traj_contact_states_;
  CIOConfig config_;
};
using CIOTrajectoryPtr = CIOTrajectory::CIOTrajectoryPtr;
