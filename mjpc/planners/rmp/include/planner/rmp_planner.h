#ifndef RMPCPP_PLANNER_PLANNER_RMP_H
#define RMPCPP_PLANNER_PLANNER_RMP_H

#include <queue>

#include "mjpc/planners/rmp/include/core/rmp_space.h"
#include "mjpc/planners/rmp/include/planner/rmp_parameters.h"
#include "mjpc/planners/rmp/include/planner/rmp_base_planner.h"
#include "mjpc/planners/rmp/include/geometry/rmp_linear_geometry.h"
#include "mjpc/planners/rmp/include/geometry/rmp_cylindrical_geometry.h"
#include "mjpc/planners/rmp/include/geometry/rmp_rotated_geometry_3d.h"
#include "mjpc/planners/rmp/include/planner/rmp_trajectory.h"
#include "mjpc/planners/rmp/include/eval/rmp_trapezoidal_integrator.h"
#include "mjpc/planners/rmp/include/util/rmp_vector_range.h"
#include "mjpc/utilities.h"

namespace rmpcpp {
/**
 * The planner class is the top-level entity that handles all planning
 * @tparam TSpace TSpace in which the world is defined (from rmpcpp/core/space)
 */
template <class TSpace>
class RMPPlanner : public RMPPlannerBase<TSpace> {
  using VectorX = typename RMPPlannerBase<TSpace>::VectorX;
  using VectorQ = typename RMPPlannerBase<TSpace>::VectorQ;
  using Matrix = typename RMPPlannerBase<TSpace>::Matrix;

 public:
  friend class TrajectoryRMP<TSpace>;
  static constexpr int dim = TSpace::dim;
  RMPPlanner() {}
  ~RMPPlanner() = default;

  using RiemannianGeometry =
#if RMP_USE_LINEAR_GEOMETRY
      LinearGeometry<TSpace::dim>;
#else
      CylindricalGeometry;
#endif
  RiemannianGeometry geometry_;

  std::vector<std::shared_ptr<RMPPolicyBase<TSpace>>> getPolicies() override
  {
    std::vector<std::shared_ptr<RMPPolicyBase<TSpace>>> policies;
    static auto default_target_policy = std::make_shared<SimpleTargetPolicy<TSpace>>();
    (*default_target_policy)(this->getGoalPos(),
                             Matrix::Identity(), 1.0, 2.0, 0.05);

    static auto default_target_policy2 = std::make_shared<SimpleTargetPolicy<TSpace>>();
    (*default_target_policy2)(this->getGoalPos(),
                              VectorQ::UnitX().asDiagonal(), 10.0, 22.0, 0.05);

    policies.push_back(default_target_policy);
    policies.push_back(default_target_policy2);
    return policies;
  }

  const std::shared_ptr<TrajectoryRMP<TSpace>> getTrajectory() const override {
    return trajectory_;  // only valid if planner has run.
  };

  bool hasTrajectory() const override { return trajectory_.operator bool(); }

  void plan(const rmpcpp::State<TSpace::dim>& start) override;

  const mjpc::Task* task_ = nullptr;
  bool checkBlocking(const VectorQ& start, const VectorQ& end) const override {
#if 0
      if ((end - start).norm() < 0.0001) {
        // Raycasting is inconsistent if they're almost on top of each other -> NOT blocking
        return false;
      }
#endif
      double start_pos[TSpace::dim];
      mju_copy(start_pos, geometry_.convertPosToX(start).data(), TSpace::dim);
      double end_pos[TSpace::dim];
      mju_copy(end_pos, geometry_.convertPosToX(end).data(), TSpace::dim);
      return task_ ? task_->CheckBlocking(start_pos, end_pos) : false;
  }

  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  // initialize data and settings
  void Initialize(mjModel* model, const mjpc::Task& task) override {
    task_ = &task;
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

    trajectory_ = std::make_unique<TrajectoryRMP<TSpace>>();
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
  void SetState(const mjpc::State& state)  override {}
  void SetStartGoal(const double* start, const double* goal) override {
    if(start) {
      this->setStartPos(geometry_.convertPosToQ(rmpcpp::vectorFromScalarArray<TSpace::dim>(start)));
    }
    if(goal) {
      this->setGoalPos(geometry_.convertPosToQ(rmpcpp::vectorFromScalarArray<TSpace::dim>(goal)));
    }
  }
  void SetStartVel(const double* vel) override {
    this->setStartVel(geometry_.convertPosToQ(rmpcpp::vectorFromScalarArray<TSpace::dim>(vel)));
  }

  // optimize nominal policy
  void OptimizePolicy(int horizon, mjpc::ThreadPool& pool)  override {
    // get nominal trajectory
    this->NominalTrajectory(horizon, pool);

    // plan
    State<TSpace::dim> startState(this->getStartPos(), this->getStartVel());

#if 1
    plan(startState);
#else
    auto starttime = std::chrono::high_resolution_clock::now();
    plan(startState);
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
    const auto* best_trajectory = static_cast<const TrajectoryRMP<TSpace>*>(BestTrajectory());
    auto currentPoint = best_trajectory->current();
#if 1
    const auto& velq = currentPoint.velocity;
    action[0] = velq[0];
    action[1] = velq[1];
#else
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
    action[0] = linear_inertia * accelq[0];
    action[1] = linear_inertia * accelq[1];
    //action[2] = linear_inertia * accelq[2];
#endif
    // Clamp controls
    mjpc::Clamp(action, model_->actuator_ctrlrange, model_->nu);
  }

  // return trajectory with best total return, or nullptr if no planning
  // iteration has completed
  const mjpc::Trajectory* BestTrajectory() override { return trajectory_.get();}

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

  TrapezoidalIntegrator<RMPPolicyBase<TSpace>, RiemannianGeometry> integrator_;
  ParametersRMP parameters_;
  std::shared_ptr<TrajectoryRMP<TSpace>> trajectory_;

  // dimensions
  int dim_state;             // state
  int dim_state_derivative;  // state derivative
  int dim_action;            // action
  int dim_sensor;            // output (i.e., all sensors)
  int dim_max;               // maximum dimension
};

// explicit instantation
template class RMPPlanner<Space<3>>;
//template class RMPPlanner<Space<2>>;
template class RMPPlanner<CylindricalSpace>;
}  // namespace rmpcpp

#endif  // RMPCPP_PLANNER_PLANNER_RMP_H
