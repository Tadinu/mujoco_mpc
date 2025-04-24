#include "mjpc/planners/lsqp/lsqp_planner.h"

#include "mjpc/utils/mjpc_math_util.h"
#include "mjpc/tasks/lsqp/lsqp.h"
#include "mjpc/planners/lsqp/lsqp_solver.h"
#include "mjpc/planners/lsqp/lsqp_collision_limit.h"

namespace mjpc {
void LsqpPlanner::Initialize(mjModel* model, const Task& task) {
  model_ = model;
  // NOTE: Not using [model->nu] here, which can be freely configured to use as many actuators as needed by users
  action_dim_ = model->nv;

  // Init task fabrics
  if (task.IsFabricsSupported()) {
    InitTaskFabrics();
  }

  // Init task Lsqp
  if (task.IsLSQPSupported()) {
    lsqp_task_ = dynamic_cast<Lsqp*>(const_cast<Task*>(&task));
    if (MjOwnerAppType::MJAPP == owner_type_) {
      InitTaskLsqp(model_, lsqp_task_->data_);
    }
    // else: MJPC: Planner running in thread, data updating jobs can only run in locked functions like [Task::TransitionLocked()]
  }

  // Init [cem_delegate_]
  if (cem_delegate_) {
    if (task.IsLSQPSupported()) {
      // Init [cem_delegate_]'s solvers
      cem_delegate_->SetPostResizeMjData([this, model]() {
        const auto& data_list = cem_delegate_->RolloutData();
        auto& solver_list = cem_delegate_->Solvers();
        const auto new_size = data_list.size();
        solver_list.reserve(new_size);
        while (solver_list.size() < new_size) {
          auto solver = std::make_shared<LsqpSolver>(model, lsqp_task_, MjOwnerAppType::MJPC);
          lsqp_task_->InitSolverConfigs(solver, data_list[solver_list.size()].get(), action_dim_);
          solver_list.emplace_back(std::move(solver));
        }
      });

      // Lsqp control callback -> embedded into [cem_delegate_]
      // Using bound lambda instead of std::bind for clarity & a bit faster
      cem_delegate_->SetControlCallback(
          [this](double* policy_action, mjData* data, const BaseSolverPtr& solver) {
            return this->LsqpControl(policy_action, data,
                                     (solver != nullptr)
                                       ? std::dynamic_pointer_cast<LsqpSolver>(solver)
                                       : nullptr);
          });
    }

    // Init [cem_delegate_], which may invoke above callbacks
    cem_delegate_->Initialize(model, task);

    // Overwrite [cem_delegate_]'s actions dim & limits
    if (task.IsLSQPSupported()) {
      cem_delegate_->SetActionDim(CEM_PARAMS_TOTAL_DIM);
      std::vector<double> limits;
      for (auto i = 0; i < CEM_PARAMS_TOTAL_DIM; ++i) {
        limits.push_back(CEM_PARAMS_LIMIT_LOWER);
        limits.push_back(CEM_PARAMS_LIMIT_UPPER);
      }
      cem_delegate_->SetActionLimits(std::move(limits));
    }
  }
}

void LsqpPlanner::InitTaskLsqp(const mjModel* model, const mjData* data) {
  // Already inited, pass
  if (!lsqp_task_ || lsqp_task_->lsqp_solver_) {
    return;
  }

  // Init [lsqp_solver_]
  lsqp_task_->InitSolver(owner_type_, action_dim_);

  // Init mocaps
  lsqp_task_->InitMocaps();
}

void LsqpPlanner::Allocate() {
  // Allocate [cem_delegate_]'s trajectory
  if (cem_delegate_) {
    cem_delegate_->InitTrajectory();
    cem_delegate_->Allocate();
  }
}

// NOTE: This can run as a callback + possibly in a rollout thread, so this function must be kept agnostic,
// thread-safe. without dynamic allocation
std::vector<double>
LsqpPlanner::LsqpControl(double* policy_action, mjData* data, const LsqpSolverPtr& solver) {
  // NOTE: !data -> invoked by [ActionFromPolicy()], else invoked by rollout thread
  if ((nullptr == lsqp_task_) || ((MjOwnerAppType::MJPC == owner_type_) && !data && !planning_on_)) {
    return InvalidPlannerControls();
  }

  // [Lsqp solver]: solve diff-ik
  // NOTE: [solver] is passed as an instance created per [LsqpControl()] to avoid dynamic allocation in threads,
  // which would cause sporadic crashes.
  return lsqp_task_->Control(policy_action, data, solver);
}

void LsqpPlanner::Traces(mjvScene* scn) {
  if (cem_delegate_) {
    cem_delegate_->Traces(scn);
  }
  if (lsqp_task_) {
    lsqp_task_->DrawTraces();
  }

#if 0
  std::vector<double> traces;
  {
    const MjpcSharedMutexLock lock(policy_mutex_);
    auto* traj = BestTrajectory();
    if (traj->trace.size() >= 6) {
      traces = traj->trace;
    }
  }

  static constexpr float GREEN[] = {0.0, 1.0, 0.0, 1.0};
  for (auto i = 0; (!traces.empty()) && (i < (traces.size() / 3) - 1); ++i) {
    AddConnector(scn ? scn : task_->scene_, mjGEOM_LINE, 5,
                       (mjtNum[]){traces[3 * i], traces[3 * i + 1], traces[3 * i + 2]},
                       (mjtNum[]){traces[3 * (i + 1)], traces[3 * (i + 1) + 1], traces[3 * (i + 1) + 2]},
                       GREEN);
  }
#endif
}

void LsqpPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  const MjpcSharedMutexLock lock(mutex_);
  if (cem_delegate_) {
    cem_delegate_->OptimizePolicy(horizon, pool);
  }
}

void LsqpPlanner::ActionFromPolicy(double* action, const double* state, double time, bool use_previous) {
  const MjpcSharedMutexLock lock(mutex_);
  if (cem_delegate_) {
    cem_delegate_->ActionFromPolicy(action, state, time, use_previous);
    // Convert [action] from [cem_delegate_]'s output space to control (data->ctrl) space
    auto ctrl = LsqpControl(action);
    if (!mjpc::AreInvalidControls(ctrl)) {
      mju_copy(action, ctrl.data(), action_dim_);
    }
  }
}
}
