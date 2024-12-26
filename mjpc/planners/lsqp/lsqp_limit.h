#pragma once

#include <stdexcept>
#include <iostream>
#include <utility>
#include <vector>

// Eigen
#include <Eigen/Dense>

// MuJoCo
#include <mujoco/mujoco.h>

// mjpc
#include "mjpc/planners/lsqp/lsqp_config.h"

namespace mjpc {
class LsqpLimit {
public:
  LsqpLimit() = default;
  virtual ~LsqpLimit() = default;

  LsqpLimit(const mjModel* model): model_(model) {
  }

  const mjModel* MjModel() const { return model_; }
  void RefreshMj(mjModel* model) { model_ = model; }
  virtual LsqpConstraint ComputeQPInequalities(const LsqpConfig& config, double dt = 1.0) const = 0;

protected:
  const mjModel* model_ = nullptr;

  constexpr int qpos_width(int jnt_type) const {
    // Example implementation, replace with actual logic as needed
    switch (jnt_type) {
      case mjJNT_HINGE:
      case mjJNT_SLIDE:
        return 1;
      case mjJNT_BALL:
        return 3;
      case mjJNT_FREE:
        return 7;
      default:
        throw std::invalid_argument("Unsupported joint type");
    }
  }

  constexpr int dof_width(int jnt_type) const {
    // Example implementation, replace with actual logic as needed
    switch (jnt_type) {
      case mjJNT_HINGE:
      case mjJNT_SLIDE:
        return 1;
      case mjJNT_BALL:
        return 3;
      case mjJNT_FREE:
        return 6;
      default:
        throw std::invalid_argument("Unsupported joint type");
    }
  }
};

using LsqpLimitPtr = std::shared_ptr<LsqpLimit>;

class LsqpPositionLimit : public LsqpLimit {
public:
  LsqpPositionLimit() = default;

  LsqpPositionLimit(const mjModel* model, double gain = 0.95, double min_distance_from_limits = 0.0)
    : LsqpLimit(model), gain_(gain) {
    if (gain <= 0.0 || gain > 1.0) {
      throw std::invalid_argument("`gain` must be in the range (0, 1]");
    }

    lower_ = Eigen::VectorXd::Constant(model_->nq, -mjMAXVAL);
    upper_ = Eigen::VectorXd::Constant(model_->nq, mjMAXVAL);

    std::vector<int> index_list;

    for (int j = 0; j < model_->njnt; ++j) {
      const int jnt_type = model_->jnt_type[j];
      const int qpos_dim = qpos_width(jnt_type);
      const double* jnt_range = &model_->jnt_range[j * 2];
      const int padr = model_->jnt_qposadr[j];

      if (jnt_type == mjJNT_FREE || !model_->jnt_limited[j]) {
        continue;
      }

      for (int i = padr; i < padr + qpos_dim; ++i) {
        lower_[i] = jnt_range[0] + min_distance_from_limits;
        upper_[i] = jnt_range[1] - min_distance_from_limits;
      }

      const int jnt_dim = dof_width(jnt_type);
      const int jnt_id = model_->jnt_dofadr[j];

      for (int i = 0; i < jnt_dim; ++i) {
        index_list.push_back(jnt_id + i);
      }
    }

    indices_ = Eigen::VectorXi::Map(index_list.data(), index_list.size());
    if (indices_.size()) {
      projection_matrix_ = Eigen::MatrixXd::Zero(indices_.size(), model_->nv);
      for (int i = 0; i < indices_.size(); ++i) {
        projection_matrix_(i, indices_[i]) = 1.0;
      }
    }
  }

  LsqpConstraint ComputeQPInequalities(const LsqpConfig& config, double dt = 1.0) const {
    if (projection_matrix_.size() == 0) {
      return {};
    }

    Eigen::VectorXd delta_q_max = Eigen::VectorXd::Zero(model_->nv);
    mj_differentiatePos(
        model_, delta_q_max.data(), dt, config.MjData()->qpos, upper_.data());

    Eigen::VectorXd delta_q_min = Eigen::VectorXd::Zero(model_->nv);
    mj_differentiatePos(
        model_, delta_q_min.data(), dt, lower_.data(), config.MjData()->qpos);

    const Eigen::VectorXd p_min = gain_ * delta_q_min(indices_);
    const Eigen::VectorXd p_max = gain_ * delta_q_max(indices_);

    const auto rows = projection_matrix_.rows();
    Eigen::MatrixXd G(2 * rows, projection_matrix_.cols());
    G.topRows(indices_.size()) = projection_matrix_;
    G.bottomRows(indices_.size()) = -projection_matrix_;

    const auto size = indices_.size();
    Eigen::VectorXd h(2 * size);
    h.head(size) = p_max;
    h.tail(size) = p_min;

    return LsqpConstraint{.G = std::move(G), .h = std::move(h)};
  }

private:
  Eigen::VectorXd lower_;
  Eigen::VectorXd upper_;
  Eigen::VectorXi indices_;
  Eigen::MatrixXd projection_matrix_;
  double gain_ = 1.0;
}; // LsqpPositionLimit


class LsqpVelocityLimit : public LsqpLimit {
public:
  LsqpVelocityLimit() = default;

  LsqpVelocityLimit(const mjModel* model,
                    const std::map<std::string, std::vector<double>>& max_joint_velocities)
    : LsqpLimit(model) {
    std::vector<double> limit_list;
    std::vector<int> index_list;

    for (const auto& [joint_name, joint_max_vels] : max_joint_velocities) {
      const int j = mj_name2id(model_, mjOBJ_JOINT, joint_name.c_str());
      const int jnt_type = model_->jnt_type[j];
      if (jnt_type == mjJNT_FREE) {
        continue;
      }
      const int vadr = model_->jnt_dofadr[j];
      const int vdim = dof_width(jnt_type);
      const std::vector<double> max_jnt_vels = (joint_max_vels.size() == 1)
                                                 ? std::vector<double>(vdim, joint_max_vels[0])
                                                 : joint_max_vels;
      assert(max_jnt_vels.size() == vdim);
      for (auto i = vadr; i < vadr + vdim; ++i) {
        index_list.push_back(i);
      }
      std::copy(max_jnt_vels.begin(), max_jnt_vels.end(), std::back_inserter(limit_list));
    }

    // [limit_list] -> [limits_]
    limits_ = Eigen::VectorXd::Map(limit_list.data(), limit_list.size());
    // [index_list] -> [indices_]
    indices_ = Eigen::VectorXi::Map(index_list.data(), index_list.size());
    if (indices_.size()) {
      projection_matrix_ = Eigen::MatrixXd::Zero(indices_.size(), model_->nv);
      for (int i = 0; i < indices_.size(); ++i) {
        projection_matrix_(i, indices_[i]) = 1.0;
      }
    }
  }

  LsqpConstraint ComputeQPInequalities(const LsqpConfig& config, double dt = 1.0) const {
    if (projection_matrix_.size() == 0) {
      return {};
    }

    const auto rows = projection_matrix_.rows();
    Eigen::MatrixXd G(2 * rows, projection_matrix_.cols());
    G.topRows(rows) = projection_matrix_;
    G.bottomRows(rows) = -projection_matrix_;

    const auto size = limits_.size();
    Eigen::VectorXd h(2 * size);
    h.head(size) = dt * limits_;
    h.tail(size) = dt * limits_;

    return LsqpConstraint{.G = std::move(G), .h = std::move(h)};
  }

protected:
  Eigen::VectorXi indices_;
  Eigen::MatrixXd limits_;
  Eigen::MatrixXd projection_matrix_;
}; // LsqpVelocityLimit
} // end namespace mjpc