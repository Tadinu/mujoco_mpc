#pragma once

// qp_solver_collection
#include <qp_solver_collection/QpSolverCollection.h>

// mjpc
#include "lsqp_relative_frame_task.h"
#include "mjpc/planners/lsqp/lsqp_common.h"
#include "mjpc/planners/lsqp/lsqp_limit.h"
#include "mjpc/planners/lsqp/lsqp_collision_limit.h"
#include "mjpc/tasks/lsqp/lsqp.h"

namespace mjpc {
static LsqpGeomIdList GetBodyGeomIds(const mjModel* model, const char* body_name) {
  const int body_id = QueryBodyId(model, body_name);
  if (body_id > 0) {
    const int geom_start = model->body_geomadr[body_id];
    auto geom_ids = LsqpGeomIdList(model->body_geomnum[body_id]);
    std::iota(geom_ids.begin(), geom_ids.end(), geom_start);
    return geom_ids;
  }
  return {};
}

static std::pair<std::vector<int>, std::vector<int>> GetFreeJointDims(const mjModel* model) {
  std::vector<int> q_ids;
  std::vector<int> v_ids;
  for (auto j = 0; j < model->njnt; ++j) {
    if (model->jnt_type[j] == mjJNT_FREE) {
      const auto qadr = model->jnt_qposadr[j];
      auto qadr_vec = std::vector<int>(7);
      std::iota(qadr_vec.begin(), qadr_vec.end(), qadr);
      q_ids.insert(q_ids.end(), qadr_vec.begin(), qadr_vec.end());

      const auto vadr = model->jnt_dofadr[j];
      auto vadr_vec = std::vector<int>(6);
      std::iota(vadr_vec.begin(), vadr_vec.end(), vadr);
      v_ids.insert(v_ids.end(), vadr_vec.begin(), vadr_vec.end());
    }
  }
  return {q_ids, v_ids};
}

static LsqpObjective ComputeQPObjective(mjData* data, const LsqpConfig& config,
                                        const std::vector<LsqpBaseTask*>& tasks, double damping) {
  Eigen::MatrixXd H = damping * Eigen::MatrixXd::Identity(config.ndofs(), config.ndofs());
  Eigen::VectorXd c = Eigen::VectorXd::Zero(config.ndofs());
  for (const auto& task : tasks) {
    const auto& [H_task, c_task] = task->ComputeQPObjective(data, config);
    H += H_task;
    c += c_task;
  }
  return LsqpObjective{.H = std::move(H), .c = std::move(c)};
}

static std::pair<std::optional<Eigen::MatrixXd>, std::optional<Eigen::VectorXd>>
ComputeQPInequalities(mjData* data, const LsqpConfig& config,
                      const std::vector<LsqpLimitPtr>& limits, double dt) {
  auto activeLimits = limits.empty()
                        ? std::vector<LsqpLimitPtr>{std::make_shared<LsqpPositionLimit>(
                            config.MjModel(), config.ndofs())}
                        : limits;

  std::vector<Eigen::MatrixXd> G_list;
  std::vector<Eigen::VectorXd> h_list;
  for (const auto& limit : activeLimits) {
    LsqpConstraint inequality = limit->ComputeQPInequalities(data, config, dt);
    if (!inequality.Inactive()) {
      if ((inequality.G.size() == 0) || (inequality.h.size()) == 0) {
        throw std::runtime_error("Invalid constraint: G or h is null.");
      }
      G_list.emplace_back(std::move(inequality.G));
      h_list.emplace_back(std::move(inequality.h));
    }
  }

  if (G_list.empty()) {
    return {std::nullopt, std::nullopt};
  }

  return {mjpc::StackEigenMatrices(G_list, false) /* vertically*/,
          mjpc::JoinEigenVectors(h_list) /*horizontally*/};
}

static Eigen::VectorXd IK_Solve(
    mjData* data,
    const LsqpConfig& config,
    const std::vector<LsqpBaseTask*>& tasks, double dt, double damping = 1e-12,
    const std::vector<LsqpLimitPtr>& limits = {}) {
  const auto qp_objective = ComputeQPObjective(data, config, tasks, damping);
  const auto [G, h] = ComputeQPInequalities(data, config, limits, dt);

  const int dim_var = config.ndofs();
  const int dim_eq = 0;
  const int dim_ineq = h.value().size();
  QpSolverCollection::QpCoeff qp_coeff;
  qp_coeff.setup(dim_var, dim_eq, dim_ineq);
  qp_coeff.obj_mat_ = qp_objective.H;
  qp_coeff.obj_vec_ = qp_objective.c;
  qp_coeff.ineq_mat_ = G.value();
  qp_coeff.ineq_vec_ = h.value();
  qp_coeff.x_min_ = Eigen::VectorXd::Constant(dim_var, -M_PI);
  qp_coeff.x_max_ = Eigen::VectorXd::Constant(dim_var, M_PI);
#if 0
  print("P", qp_coeff.obj_mat_.rows(), qp_coeff.obj_mat_.cols(), qp_coeff.obj_mat_);
  print("q", qp_coeff.obj_vec_.rows(), qp_coeff.obj_vec_.cols(), qp_coeff.obj_vec_);
  print("G", qp_coeff.ineq_mat_.rows(), qp_coeff.ineq_mat_.cols(), qp_coeff.ineq_mat_);
  print("h", qp_coeff.ineq_vec_.rows(), qp_coeff.ineq_vec_.cols(), qp_coeff.ineq_vec_);
#endif
  const Eigen::VectorXd dq = config.QpSolver()->solve(qp_coeff);
#if MJPC_PLANNER_LSQP_DEBUG
  if (!dq.isZero()) {
    print("[IK_Solve]-dq: ", dq.transpose());
  }
#endif
  return dq / dt;
}
}
