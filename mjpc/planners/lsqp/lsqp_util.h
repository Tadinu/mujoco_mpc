#pragma once

// qp_solver_collection
#include <qp_solver_collection/QpSolverCollection.h>

// daqp
#include <daqp/daqp.h>
#include <daqp/api.h>

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

static LsqpObjective ComputeQPObjective(const mjData* data, const LsqpConfig& config,
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
ComputeQPInequalities(const mjData* data, const LsqpConfig& config,
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
    const mjData* data,
    const LsqpConfig& config,
    const std::vector<LsqpBaseTask*>& tasks, double dt, double damping = 1e-12,
    const std::vector<LsqpLimitPtr>& limits = {}) {
  const auto qp_objective = ComputeQPObjective(data, config, tasks, damping);
  const auto [G, h] = ComputeQPInequalities(data, config, limits, dt);
  const int dim_var = config.ndofs();
  const int dim_eq = 0;
  const int dim_ineq = h.value().size();
  const int dim_ineq_eq = dim_eq + dim_ineq;

#if 1
  print("P", qp_objective.H.rows(), qp_objective.H.cols(), qp_objective.H);
  print("q", qp_objective.c.rows(), qp_objective.c.cols(), qp_objective.c);
  print("G", G.value().rows(), G.value().cols(), G.value());
  print("h", h.value().rows(), h.value().cols(), h.value());
#endif

#if MJPC_LSQP_USE_QP_SOLVER_COLLECTION
  QpSolverCollection::QpCoeff qp_coeff;
  qp_coeff.setup(dim_var, dim_eq, dim_ineq);
  qp_coeff.obj_mat_ = qp_objective.H;
  qp_coeff.obj_vec_ = qp_objective.c;
  qp_coeff.ineq_mat_ = G.value();
  qp_coeff.ineq_vec_ = h.value();
  qp_coeff.x_min_ = Eigen::VectorXd::Constant(dim_var, -M_PI);
  qp_coeff.x_max_ = Eigen::VectorXd::Constant(dim_var, M_PI);
  const Eigen::VectorXd dq = config.QpSolver()->solve(qp_coeff);
#elif MJPC_LSQP_USE_DAQP_SOLVER
  // Define the problem
  std::vector<double> H(qp_objective.H.size(), 0);
  mjpc::ArrayFromEigenMatrix(qp_objective.H, H.data());
  std::vector<double> G_arr(G.value().size(), 0);
  mjpc::ArrayFromEigenMatrix(G.value(), G_arr.data());
  auto bupper = std::vector<double>(dim_var, M_PI);
  bupper = mjpc::ChainCollections<double>(bupper, limits[0]->Upper()); // h.value().tail(dim_ineq / 2)
  auto blower = std::vector<double>(dim_var, -M_PI);
  blower = mjpc::ChainCollections<double>(blower, limits[0]->Lower()); // h.value().head(dim_ineq / 2)
  auto sense = std::vector<int>(dim_ineq_eq, 0);
  memset(sense.data() + dim_ineq, 5, dim_eq * sizeof(int));
  DAQPProblem qp = {dim_var, dim_ineq_eq /* No of constraints (general + simple) */,
                    0 /* No of simple bounds */,
                    H.data(),
                    const_cast<double*>(qp_objective.c.data()),
                    G_arr.data(),
                    bupper.data(),
                    blower.data(),
                    sense.data()};

  // Settings
  auto res_x = std::vector<double>(dim_var, 0);
  auto res_lam = std::vector<double>(dim_ineq_eq, 0);
  DAQPResult result = {res_x.data(), res_lam.data(), 0, 0, 0, 0, 0, 0};
  DAQPSettings settings;
  daqp_default_settings(&settings); // Populate settings with default values
  settings.iter_limit = 2000;

  // Solve
  daqp_quadprog(&result, &qp, &settings);
  const Eigen::VectorXd dq =
      result.x ? mjpc::ArrayToEigen(result.x, dim_var) : Eigen::VectorXd::Constant(dim_var, mjMAXVAL + 1);
#endif

#if MJPC_PLANNER_LSQP_DEBUG
  if (!dq.isZero()) {
    print("[IK_Solve]-dq: ", dq.transpose());
  }
#endif

  assert(dq.size() == config.ndofs());
  return dq / dt;
}
}
