#pragma once

// qp_solver_collection
#include <qp_solver_collection/QpSolverCollection.h>

// mjpc
#include "lsqp_relative_frame_task.h"
#include "mjpc/planners/lsqp/lsqp_common.h"
#include "mjpc/planners/lsqp/lsqp_config.h"
#include "mjpc/planners/lsqp/lsqp_limit.h"
#include "mjpc/planners/lsqp/lsqp_base_task.h"
#include "mjpc/tasks/lsqp/lsqp.h"

namespace mjpc {
static std::pair<std::vector<int>, std::vector<int>> GetFreeJointDims(const mjModel* model) {
  std::vector<int> q_ids;
  std::vector<int> v_ids;
  for (auto j = 0; j < model->njnt; ++j) {
    if (model->jnt_type[j] == mjJNT_FREE) {
      const auto qadr = model->jnt_qposadr[j];
      const auto vadr = model->jnt_dofadr[j];
      const auto qadr_vec = std::vector<int>(qadr, qadr + 7);
      const auto vadr_vec = std::vector<int>(vadr, vadr + 6);
      std::copy_n(qadr_vec.begin(), 7, std::back_inserter(q_ids));
      std::copy_n(vadr_vec.begin(), 6, std::back_inserter(v_ids));
    }
  }
  return {q_ids, v_ids};
}

static LsqpObjective ComputeQPObjective(const LsqpConfig& config,
                                        const std::vector<LsqpBaseTask*>& tasks, double damping) {
  Eigen::MatrixXd H = damping * Eigen::MatrixXd::Identity(config.nv(), config.nv());
  Eigen::VectorXd c = Eigen::VectorXd::Zero(config.nv());
  for (const auto& task : tasks) {
    const auto& [H_task, c_task] = task->ComputeQPObjective(config);
    H += H_task;
    c += c_task;
  }
  return LsqpObjective{.H = std::move(H), .c = std::move(c)};
}

static std::pair<std::optional<Eigen::MatrixXd>, std::optional<Eigen::VectorXd>>
ComputeQPInequalities(const LsqpConfig& config,
                      const std::vector<LsqpLimitPtr>& limits, double dt) {
  auto activeLimits = limits.empty()
                        ? std::vector<LsqpLimitPtr>{std::make_shared<LsqpPositionLimit>(config.MjModel())}
                        : limits;

  std::vector<Eigen::MatrixXd> G_list;
  std::vector<Eigen::VectorXd> h_list;
  for (const auto& limit : activeLimits) {
    LsqpConstraint inequality = limit->ComputeQPInequalities(config, dt);
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
    const LsqpConfig& config,
    const std::vector<LsqpBaseTask*>& tasks, double dt, double damping = 1e-12,
    const std::vector<LsqpLimitPtr>& limits = {}) {
  const auto qp_objective = ComputeQPObjective(config, tasks, damping);
  const auto [G, h] = ComputeQPInequalities(config, limits, dt);

  const int dim_var = config.nv();
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
  print("P", qp_coeff.obj_mat_);
  print("q", qp_coeff.obj_vec_);
  print("G", qp_coeff.ineq_mat_);
  print("h", qp_coeff.ineq_vec_);
#endif
  const auto qp_solver = QpSolverCollection::allocateQpSolver(QpSolverCollection::QpSolverType::QuadProg);
  const Eigen::VectorXd dq = qp_solver->solve(qp_coeff);
#if MJPC_PLANNER_LSQP_DEBUG
  if (!dq.isZero()) {
    print("[IK_Solve]-dq: ", dq.transpose());
  }
#endif
  return dq / dt;
}
}