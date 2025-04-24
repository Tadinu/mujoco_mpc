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
// Ref: https://github.com/kevinzakka/mink/blob/main/mink/limits/limit.py
class LsqpLimit {
public:
  LsqpLimit() = default;
  virtual ~LsqpLimit() = default;

  LsqpLimit(const mjModel* model, int ndofs): model_(model), ndofs_(ndofs) {
  }

  const mjModel* MjModel() const { return model_; }
  int ndofs() const { return ndofs_; }
  int nv() const { return model_->nv; }
  int nq() const { return model_->nq; }
  virtual LsqpConstraint ComputeQPInequalities(const mjData* data, const LsqpConfig& config,
                                               double dt = 1.0) const = 0;

  virtual Eigen::VectorXd Lower() const { return {}; }
  virtual Eigen::VectorXd Upper() const { return {}; }

protected:
  const mjModel* model_ = nullptr;
  int ndofs_ = 0;

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

// Refs:
// [kevinzakka]: https://github.com/kevinzakka/mink/blob/main/mink/limits/configuration_limit.py
// [dm_robotics] : https://github.com/google-deepmind/dm_robotics/blob/main/cpp/controllers/lsqp/include/dm_robotics/controllers/lsqp/joint_position_limit_constraint.h
class LsqpPositionLimit : public LsqpLimit {
public:
  LsqpPositionLimit() = default;

  LsqpPositionLimit(const mjModel* model, int ndofs, double gain = 0.95,
                    double min_distance_from_limits = 0.0)
    : LsqpLimit(model, ndofs), gain_(gain) {
    if (gain <= 0.0 || gain > 1.0) {
      throw std::invalid_argument("`gain` must be in the range (0, 1]");
    }

    lower_ = Eigen::VectorXd::Constant(nq(), -mjMAXVAL);
    upper_ = Eigen::VectorXd::Constant(nq(), mjMAXVAL);

    std::vector<int> index_list;
    for (int j = 0; j < model_->njnt; ++j) {
      const int jnt_type = model_->jnt_type[j];
      if (jnt_type == mjJNT_FREE || !model_->jnt_limited[j]) {
        continue;
      }

      const int qpos_dim = qpos_width(jnt_type);
      const double* jnt_range = &model_->jnt_range[j * 2];
      const int padr = model_->jnt_qposadr[j];
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

    if (!index_list.empty()) {
      assert(index_list.size() <= ndofs_);
      indices_ = Eigen::VectorXi::Map(index_list.data(), index_list.size());
      projection_matrix_ = Eigen::MatrixXd::Zero(index_list.size(), ndofs_);
      for (int i = 0; i < indices_.size(); ++i) {
        projection_matrix_(i, indices_[i]) = 1.0;
      }
    }
  }

  Eigen::VectorXd Lower() const override { return lower_(indices_); }
  Eigen::VectorXd Upper() const override { return upper_(indices_); }

  LsqpConstraint ComputeQPInequalities(const mjData* data, const LsqpConfig& config, double dt = 1.0) const {
    if (projection_matrix_.size() == 0) {
      return {};
    }

    // NOTE: Ignore timestep!
    dt = 1.0;

    // NOTE: Don't use mj_differentiatePos(), which loops over all joints, assuming inputs as c-arrays also hosting all such joints
    const auto* qpos = data->qpos;
    Eigen::VectorXd dq_min(lower_.size());
    mju_sub(dq_min.data(), qpos, lower_.data(), lower_.size());
    mju_scl(dq_min.data(), dq_min.data(), 1 / dt, lower_.size());

    Eigen::VectorXd dq_max(upper_.size());
    mju_sub(dq_max.data(), upper_.data(), qpos, upper_.size());
    mju_scl(dq_max.data(), dq_max.data(), 1 / dt, upper_.size());

    const Eigen::VectorXd p_min = gain_ * dq_min(indices_);
    const Eigen::VectorXd p_max = gain_ * dq_max(indices_);

    // https://kevinzakka.github.io/mink/derivations.html
    // dq = v * dt
    // q_min <= q + dq <= q_max
    //  -dq <= (q - q_min)
    //   dq <= (q_max - q)
    // [G*dq <= h], G = [1, -1]^T, h = [q_max - q, q - q_min]^T
    const auto rows = projection_matrix_.rows();
    Eigen::MatrixXd G(2 * rows, projection_matrix_.cols());
    G.topRows(indices_.size()) = -projection_matrix_; // G_min
    G.bottomRows(indices_.size()) = projection_matrix_; // G_max

    // NOTE: p_min, p_max are fed into [h] following the structure of [G]
    const auto size = indices_.size();
    Eigen::VectorXd h(2 * size);
    h.head(size) = p_min;
    h.tail(size) = p_max;

    return LsqpConstraint{.G = std::move(G), .h = std::move(h)};
  }

private:
  Eigen::VectorXd lower_;
  Eigen::VectorXd upper_;
  Eigen::VectorXi indices_;
  Eigen::MatrixXd projection_matrix_;
  double gain_ = 1.0;
}; // LsqpPositionLimit


// Refs:
// [kevinzakka]: https://github.com/kevinzakka/mink/blob/main/mink/limits/velocity_limit.py
// [dm_robotics]: https://github.com/google-deepmind/dm_robotics/blob/main/cpp/controllers/lsqp/include/dm_robotics/controllers/lsqp/joint_velocity_filter.h
class LsqpVelocityLimit : public LsqpLimit {
public:
  LsqpVelocityLimit() = default;

  LsqpVelocityLimit(const mjModel* model, int ndofs,
                    const std::map<std::string, std::vector<double>>& max_joint_velocities)
    : LsqpLimit(model, ndofs) {
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
    if (!limit_list.empty()) {
      assert(ndofs_ <= limit_list.size());
      limits_ = Eigen::VectorXd::Map(limit_list.data(), ndofs_);
    }

    // [index_list] -> [indices_]
    if (!index_list.empty()) {
      assert(ndofs_ <= index_list.size());
      indices_ = Eigen::VectorXi::Map(index_list.data(), ndofs_);
      projection_matrix_ = Eigen::MatrixXd::Zero(ndofs_, ndofs_);
      for (int i = 0; i < indices_.size(); ++i) {
        projection_matrix_(i, indices_[i]) = 1.0;
      }
    }
  }

  LsqpConstraint ComputeQPInequalities(const mjData* data, const LsqpConfig& config, double dt = 1.0) const {
    if (projection_matrix_.size() == 0) {
      return {};
    }

    // https://kevinzakka.github.io/mink/derivations.html
    // [G*dq <= h]
    const auto rows = projection_matrix_.rows();
    Eigen::MatrixXd G(2 * rows, projection_matrix_.cols());
    G.topRows(rows) = -projection_matrix_;
    G.bottomRows(rows) = projection_matrix_;

    // -limits_ <= dq/dt <= limits
    const auto size = limits_.size();
    const auto max_limits = dt * limits_;
    Eigen::VectorXd h(2 * size);
    h.head(size) = max_limits;
    h.tail(size) = max_limits;

    return LsqpConstraint{.G = std::move(G), .h = std::move(h)};
  }

protected:
  Eigen::VectorXi indices_;
  Eigen::MatrixXd limits_;
  Eigen::MatrixXd projection_matrix_;
}; // LsqpVelocityLimit
} // end namespace mjpc