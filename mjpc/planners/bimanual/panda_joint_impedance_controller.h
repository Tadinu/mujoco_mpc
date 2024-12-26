// Copyright (c) 2020 Lehrstuhl für Robotik und Sysstemintelligenz, TU München
#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

// mjpc
#include "mjpc/task.h"


namespace mjpc {
template <int N = 7>
class PandaJointImpedanceController {
  using VectorNd = Eigen::Matrix<double, N, 1>;
  using MatrixNd = Eigen::Matrix<double, N, N>;

public:
  explicit PandaJointImpedanceController(mjModel* model, mjData* data, Task* task):
    mj_model_(model), mj_data_(data), mj_task_(task) {
  }

  mjModel* mj_model_ = nullptr;
  mjData* mj_data_ = nullptr;
  Task* mj_task_ = nullptr;
  VectorNd control(const VectorNd& q_d, const VectorNd& qD_d = VectorNd::Zero());
  void init(const std::string& first_joint_name, const std::string& endtip_name);
  double limitQD(VectorNd qD_d, const double& v_cart);
  void setStiffnessScale(const double& stiffness_scale);

  VectorNd getQ() const {
    auto q = mj_task_->QueryJointPositions(N, first_joint_name_);
    return Eigen::Map<VectorNd>(q.data(), N);
  }

  VectorNd getQD() const {
    auto qd = mj_task_->QueryJointVels(N, first_joint_name_);
    return Eigen::Map<VectorNd>(qd.data(), N);
  }

  VectorNd getTauExtHat(bool filtering = true) const {
    assert(N<=mj_model_->nv);
    VectorNd tau_measured = VectorNd::Zero();
    mju_add(tau_measured.data(), &mj_data_->qfrc_applied[first_dof_id_],
            &mj_data_->qfrc_constraint[first_dof_id_], N);
    mju_addTo(tau_measured.data(), &mj_data_->qfrc_bias[first_dof_id_], N);
    mju_subFrom(tau_measured.data(), &mj_data_->qfrc_actuator[first_dof_id_], N);

    if (filtering) {
      const auto low_pass_filter = [](mjtNum* data, double alpha = 0.1) {
        VectorNd filtered = VectorXd::Zero(N);
        for (auto i = 0; i < N; ++i) {
          filtered[i] = alpha * data[i] + (1 - alpha) * ((i > 0) ? filtered[i - 1] : data[i]);
        }
        return filtered;
      };
      return low_pass_filter(tau_measured.data(), N);
    } else {
      return tau_measured;
    }
  }

private:
  int first_dof_id_ = -1;
  std::string first_joint_name_;
  std::string endtip_name_;
  double stiffness_scale_ = 1;
  MatrixNd jointDampingDesign(const MatrixNd& stiffness,
                              const MatrixNd& damping_ratio,
                              const MatrixNd& inertia);
};
} // namespace mjpc
