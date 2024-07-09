#ifndef RMP_PLANNER_PLANNER_RMP_H
#define RMP_PLANNER_PLANNER_RMP_H

#include <queue>
#include <utility>

#include "mjpc/planners/rmp/include/core/rmp_parameters.h"
#include "mjpc/planners/rmp/include/eval/rmp_trapezoidal_integrator.h"
#include "mjpc/planners/rmp/include/geometry/rmp_cylindrical_geometry.h"
#include "mjpc/planners/rmp/include/geometry/rmp_linear_geometry.h"
#include "mjpc/planners/rmp/include/geometry/rmp_rotated_geometry_3d.h"
#include "mjpc/planners/rmp/include/planner/rmp_base_planner.h"
#include "mjpc/planners/rmp/include/planner/rmp_trajectory.h"
#include "mjpc/planners/rmp/include/policies/rmp_base_policy.h"
#include "mjpc/planners/rmp/include/policies/rmp_raycasting_policy.h"
#include "mjpc/planners/rmp/include/policies/rmp_simple_target_policy.h"
#include "mjpc/planners/rmp/include/util/rmp_util.h"
#include "mjpc/utilities.h"

namespace rmp {
/**
 * The planner class is the top-level entity that handles all planning
 * @tparam TSpace TSpace in which the world is defined (from rmp/core/space)
 */
template <class TSpace>
class RMPPlanner : public RMPPlannerBase<TSpace> {
  using VectorX = typename RMPPlannerBase<TSpace>::VectorX;
  using VectorQ = typename RMPPlannerBase<TSpace>::VectorQ;
  using Matrix = typename RMPPlannerBase<TSpace>::Matrix;
  using StateX = typename RMPPlannerBase<TSpace>::StateX;

 public:
  friend class RMPTrajectory<TSpace>;
  static constexpr int dim = TSpace::dim;
  RMPPlanner() = default;
  explicit RMPPlanner(RMPConfigs configs): configs_(std::move(configs)) {}
  ~RMPPlanner() override = default;

  std::vector<std::shared_ptr<RMPPolicyBase<TSpace>>> getPolicies() override
  {
    std::vector<std::shared_ptr<RMPPolicyBase<TSpace>>> policies;

    const auto goal_pos = vectorFromScalarArray<TSpace::dim>(task_->GetGoalPos());

    // Target attractor policies (For the collision policies -> [RaycastingPolicyConfigs]
    // Ref: [Table I]
    static auto simple_target_policy = std::make_shared<SimpleTargetPolicy<TSpace>>();
    (*simple_target_policy)(goal_pos,
                            Matrix::Identity(), 0.8, 1.6, 1.0);

    static auto simple_target_policy1 = std::make_shared<SimpleTargetPolicy<TSpace>>();
    (*simple_target_policy1)(goal_pos,
                            Matrix::Identity(), 1.6, 3.2, 1.0);
    static auto simple_target_policy2 = std::make_shared<SimpleTargetPolicy<TSpace>>();
    (*simple_target_policy2)(goal_pos,
                             VectorQ::UnitX().asDiagonal(), 20.0, 44.0, 20);
    static auto simple_target_policy3 = std::make_shared<SimpleTargetPolicy<TSpace>>();
    (*simple_target_policy3)(goal_pos,
                             VectorQ::UnitY().asDiagonal(), 20.0, 44.0, 20);
    policies.push_back(simple_target_policy);
    policies.push_back(simple_target_policy1);
    policies.push_back(simple_target_policy2);
    policies.push_back(simple_target_policy3);

    // Collision avoidance policy
#if RMP_USE_RMP_COLLISION_POLICY
    static auto rmp_collision_policy =
        RMPPolicyBase<TSpace>::template MakePolicy<
            RaycastingPolicy<TSpace>, ERMPPolicyType::RAYCASTING>();
    policies.push_back(rmp_collision_policy);
#endif

    for (auto& policy : policies) {
      policy->scene_ = task_->scene_;
    }
    return policies;
  }

  std::shared_ptr<RMPTrajectory<TSpace>> getTrajectory() const override {
    return trajectory_;  // only valid if planner has run.
  };

  bool hasTrajectory() const override { return trajectory_.operator bool(); }

  // Plan a path from start -> goal
  void plan() override;

  mjpc::Task* task_ = nullptr;
  bool checkBlocking(const VectorQ& start, const VectorQ& end) override {
#if 0
      if ((end - start).norm() < 0.0001) {
        // Raycasting is inconsistent if they're almost on top of each other -> NOT blocking
        return false;
      }
#endif
      double start_pos[TSpace::dim];
      mju_copy(start_pos, this->geometry_.convertPosToX(start).data(), TSpace::dim);
      double end_pos[TSpace::dim];
      mju_copy(end_pos, this->geometry_.convertPosToX(end).data(), TSpace::dim);
      return task_ ? task_->CheckBlocking(start_pos, end_pos) : false;
  }

  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  // initialize data and settings
  void Initialize(mjModel* model, const mjpc::Task& task) override {
    task_ = const_cast<mjpc::Task*>(&task);
    integrator_.task_ = task_;
    model_ = model;
    data_ = task_->data_;

    // dimensions
    dim_state = model->nq + model->nv + model->na;  // state dimension
    dim_state_derivative =
        2 * model->nv + model->na;    // state derivative dimension
    dim_action = model->nu;           // action dimension
    dim_sensor = model->nsensordata;  // number of sensor values
    dim_max =
        mju_max(mju_max(mju_max(dim_state, dim_state_derivative), dim_action),
                model->nuser_sensor);

    trajectory_ = std::make_shared<RMPTrajectory<TSpace>>();
    trajectory_->setMaxLength(configs_.max_length);
  }

  // allocate memory
  void Allocate()  override {
    trajectory_->Initialize(dim_state, dim_action, task_->num_residual,
                            task_->num_trace, 1);
    trajectory_->Allocate(1);
  }

  // reset memory to zeros
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr)  override {}

  // set state
  void SetState(const mjpc::State& state) override {}

  VectorQ GetStartPosQ() const
  {
    return this->geometry_.convertPosToQ(vectorFromScalarArray<TSpace::dim>(task_->GetStartPos()));
  }

  VectorQ GetStartVelQ() const
  {
    return this->geometry_.convertPosToQ(vectorFromScalarArray<TSpace::dim>(task_->GetStartVel()));
  }

  VectorQ GetGoalPosQ() const
  {
    return this->geometry_.convertPosToQ(vectorFromScalarArray<TSpace::dim>(task_->GetGoalPos()));
  }

  // optimize nominal policy
  void OptimizePolicy(int horizon, mjpc::ThreadPool& pool)  override {
    const std::shared_lock<std::shared_mutex> lock(policy_mutex_);
    // get nominal trajectory
    this->NominalTrajectory(horizon, pool);

    // plan
#if 1
    plan();
#else
    auto starttime = std::chrono::high_resolution_clock::now();
    plan();
    auto endtime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        endtime - starttime);
    //double duration_s = double(duration.count()) / 1E6;

    std::string success = this->success() ? "Success: " : "Failure: ";
    std::cout << success << double(duration.count()) / 1000.0 << "ms"
              << std::endl;
#endif
  }

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, mjpc::ThreadPool& pool) override {
    //auto nominal_policy = [](double* action, const double* state, double time) {
    //};

    // rollout nominal policy
    //trajectory_->Rollout(nominal_policy, task_, model_,
    //                     data_[mjpc::ThreadPool::WorkerId()].get(), state.data(),
    //                     time, mocap.data(), userdata.data(), horizon);
  }

  // set action from policy
  void ActionFromPolicy(double* action, const double* state,
                        double time, bool use_previous = false) override {
    const std::shared_lock<std::shared_mutex> lock(policy_mutex_);
    const auto* best_trajectory = static_cast<const RMPTrajectory<TSpace>*>(BestTrajectory());
    auto currentPoint = best_trajectory->current();
#if RMP_USE_ACTUATOR_VELOCITY
    const auto& velq = currentPoint.velocity;
    action[0] = RMP_KV * velq[0];
    action[1] = RMP_KV * velq[1];
#elif RMP_USE_ACTUATOR_MOTOR
    const auto& accelq = currentPoint.acceleration;
    const auto pointmass_id = task_->GetTargetObjectId();
    const auto pointmass = model_->body_mass[pointmass_id];
    const auto& linear_inertia = pointmass;

    //auto body_inertia = model_->body_inertia[3*pointmass_id];
    // auto body_extent = 0.01; //mju_sqrt(body_inertia / pointmass);
    //auto rotationalInertia = 0.4 * pointmass * mju_pow(body_extent, 2);
    //mjtNum I1 = model_->body_inertia[3*pointmass_id+0];
    //mjtNum I2 = model_->body_inertia[3*pointmass_id+1];
    //mjtNum I3 = model_->body_inertia[3*pointmass_id+2];

    //NOTE: Depending on which dofs the action ctrl is configured in xml, eg: linear movement only uses linear_inertia only
    action[0] = RMP_FORCE_GAIN * linear_inertia * accelq[0];
    action[1] = RMP_FORCE_GAIN * linear_inertia * accelq[1];
    //action[2] = linear_inertia * accelq[2];
#endif
    // Clamp controls
    mjpc::Clamp(action, model_->actuator_ctrlrange, model_->nu);
  }

  // return trajectory with best total return, or nullptr if no planning
  // iteration has completed
  const mjpc::Trajectory* BestTrajectory() override { return trajectory_.get();}
  void DrawTrajectoryCurrent(const mjpc::Trajectory* trajectory, mjvScene* scn = nullptr, const float* color = nullptr);
  void DrawTrajectory(const mjpc::Trajectory* trajectory, mjvScene* scn = nullptr, const float* color = nullptr);

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override;

  // planner-specific GUI elements
  void GUI(mjUI& ui) override {}

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
             int planner_shift, int timer_shift, int planning,
             int* shift) override {}

  // return number of parameters optimized by planner
  int NumParameters() override {return 0;}
 private:
  void integrate();
  mutable std::shared_mutex policy_mutex_;

  TrapezoidalIntegrator<RMPPolicyBase<TSpace>, typename RMPPlannerBase<TSpace>::RiemannianGeometry> integrator_;
  RMPConfigs configs_;
  std::shared_ptr<RMPTrajectory<TSpace>> trajectory_;

  // dimensions
  int dim_state = 0;             // state
  int dim_state_derivative = 0;  // state derivative
  int dim_action = 0;            // action
  int dim_sensor = 0;            // output (i.e., all sensors)
  int dim_max = 0;               // maximum dimension
};
}  // namespace rmp

#endif  // RMP_PLANNER_PLANNER_RMP_H
