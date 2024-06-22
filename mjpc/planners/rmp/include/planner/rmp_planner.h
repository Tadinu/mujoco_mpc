#ifndef RMPCPP_PLANNER_PLANNER_RMP_H
#define RMPCPP_PLANNER_PLANNER_RMP_H

#include <queue>

#include "mjpc/planners/rmp/include/core/rmp_space.h"
#include "mjpc/planners/rmp/include/planner/rmp_parameters.h"
#include "mjpc/planners/rmp/include/planner/rmp_base_planner.h"
#include "mjpc/planners/rmp/include/planner/rmp_trajectory.h"
#include "mjpc/planners/rmp/include/util/rmp_vector_range.h"
#include "mjpc/utilities.h"

namespace rmpcpp {
/**
 * The planner class is the top-level entity that handles all planning
 * @tparam TSpace TSpace in which the world is defined (from rmpcpp/core/space)
 */
template <class TSpace>
class RMPPlanner : public RMPPlannerBase<TSpace> {
  using Vector = Eigen::Matrix<double, TSpace::dim, 1>;
  using Matrix = Eigen::Matrix<double, TSpace::dim, TSpace::dim>;

 public:
  friend class TrajectoryRMP<TSpace>;
  static constexpr int dim = TSpace::dim;
  RMPPlanner() {}
  ~RMPPlanner() = default;

  std::vector<std::shared_ptr<RMPPolicyBase<TSpace>>> getPolicies() override
  {
    std::vector<std::shared_ptr<RMPPolicyBase<TSpace>>> policies;
    static auto default_target_policy_ = std::make_shared<SimpleTargetPolicy<TSpace>>();
    (*default_target_policy_)(this->getGoalPos(),
                              Matrix::Identity(), 1.0, 2.0, 0.05);

    static auto default_target_policy2_ = std::make_shared<SimpleTargetPolicy<TSpace>>();
    (*default_target_policy2_)(this->getGoalPos(),
                               Vector::UnitX().asDiagonal(), 10.0, 22.0, 0.05);

    policies.push_back(default_target_policy_);
    policies.push_back(default_target_policy2_);
    return policies;
  }

  const std::shared_ptr<TrajectoryRMP<TSpace>> getTrajectory() const override {
    return trajectory_;  // only valid if planner has run.
  };

  bool hasTrajectory() const override { return trajectory_.operator bool(); }

  void plan(const rmpcpp::State<TSpace::dim>& start,
            const Vector& goal) override;

  const mjpc::Task* task_ = nullptr;
  bool checkBlocking(const Vector& s1, const Vector& s2) const override {
      if ((s2 - s1).norm() < 0.0001) {
        // Raycasting is inconsistent if they're almost on top of each other -> NOT blocking
        return false;
      }
      double pos1[TSpace::dim];
      memcpy(pos1, s1.data(), sizeof(double) * TSpace::dim);
      double pos2[TSpace::dim];
      memcpy(pos2, s2.data(), sizeof(double) * TSpace::dim);
      return task_ ? task_->checkCollision(pos1) || task_->checkCollision(pos2): true;
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
      this->setStartPos(rmpcpp::vectorFromScalarArray<TSpace::dim>(start));
    }
    if(goal) {
      this->setGoalPos(rmpcpp::vectorFromScalarArray<TSpace::dim>(goal));
    }
  }
  void SetStartVel(const double* vel) override {
    this->setStartVel(rmpcpp::vectorFromScalarArray<TSpace::dim>(vel));
  }

  // optimize nominal policy
  void OptimizePolicy(int horizon, mjpc::ThreadPool& pool)  override {
    // get nominal trajectory
    this->NominalTrajectory(horizon, pool);

    // plan
    Vector startPos = this->getStartPos();
    Vector goalPos = this->getGoalPos();
    State<TSpace::dim> startState(startPos, this->getStartVel());

#if 1
    plan(startState, goalPos);
#else
    auto starttime = std::chrono::high_resolution_clock::now();
    plan(startState, goalPos);
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
    const auto currentPoint = static_cast<const TrajectoryRMP<TSpace>*>(BestTrajectory())
                                  ->current();
#if 1
    const auto& vel = currentPoint.velocity;
    action[0] = vel[0];
    action[1] = vel[1];
#else
    const auto& acceleration = currentPoint.acceleration;
    auto pointmass_id = mj_name2id(model_, mjOBJ_BODY, "pointmass");
    auto pointmass = model_->body_mass[pointmass_id];

    auto body_inertia = model_->body_inertia[3*pointmass_id];
    auto body_extent = mju_sqrt(body_inertia / pointmass);
    auto inertia = 0.4*pointmass * mju_pow(body_extent, 2);
      //mjtNum I1 = model_->body_inertia[3*pointmass_id+0];
      //mjtNum I2 = model_->body_inertia[3*pointmass_id+1];
      //mjtNum I3 = model_->body_inertia[3*pointmass_id+2];

      //mjtNum res[3];
      //mju_cross(res, {vel[0], vel[1], vel[2]}, {I1, I2, I3});
      action[0] = 1000*inertia * acceleration[0];
      action[1] = 1000*inertia * acceleration[1];
      //action[2] = Iyy * acceleration[2];
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
template class RMPPlanner<Space<2>>;
template class RMPPlanner<CylindricalSpace>;
}  // namespace rmpcpp

#endif  // RMPCPP_PLANNER_PLANNER_RMP_H
