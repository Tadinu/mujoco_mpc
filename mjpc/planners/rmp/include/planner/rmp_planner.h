#ifndef RMPCPP_PLANNER_PLANNER_RMP_H
#define RMPCPP_PLANNER_PLANNER_RMP_H

#include <queue>

#include "mjpc/planners/rmp/include/core/rmp_space.h"
#include "mjpc/planners/rmp/include/planner/rmp_parameters.h"
#include "mjpc/planners/rmp/include/planner/rmp_base_planner.h"
#include "mjpc/planners/rmp/include/planner/rmp_trajectory.h"
#include "mjpc/utilities.h"

namespace rmpcpp {
/**
 * The planner class is the top-level entity that handles all planning
 * @tparam TSpace TSpace in which the world is defined (from rmpcpp/core/space)
 */
template <class TSpace>
class RMPPlanner : public RMPPlannerBase<TSpace> {
  using Vector = Eigen::Matrix<double, TSpace::dim, 1>;

 public:
  friend class TrajectoryRMP<TSpace>;
  static constexpr int dim = TSpace::dim;
  RMPPlanner() {}
  ~RMPPlanner() = default;

  std::shared_ptr<RMPPolicyBase<TSpace>> default_target_policy_ =
    std::make_shared<SimpleTargetPolicy<TSpace>>(Eigen::Vector3d{2.0, M_PI_2, 0.0},
                                                 Eigen::Matrix3d::Identity(), 1.0, 2.0, 0.05);
  std::shared_ptr<RMPPolicyBase<TSpace>> default_target_policy2_ =
    std::make_shared<SimpleTargetPolicy<TSpace>>(Eigen::Vector3d{3.0, 0, 0.0},
                                                 Eigen::Vector3d({1.0, 0.0, 0.0}).asDiagonal(), 10.0, 22.0, 0.05);
  std::vector<std::shared_ptr<RMPPolicyBase<TSpace>>> getPolicies() override
  {
    std::vector<std::shared_ptr<RMPPolicyBase<TSpace>>> policies;
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
  bool checkMotion(const Vector& s1, const Vector& s2) const override {
      return task_ ? !task_->checkCollision((double[]){s2[0], s2[1], s2[2]}) : false;
  }

  mjModel* model_ = nullptr;
  // initialize data and settings
  void Initialize(mjModel* model, const mjpc::Task& task) override {
    task_ = &task;
    model_ = model;

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
      this->setStartPos({start[0], start[1], start[2]});
    }
    if(goal) {
      this->setGoalPos({goal[0], goal[1], goal[2]});
    }
  }
  void SetStartVel(const double* start_vel) override {
    if(start_vel) {
      this->setStartVel({start_vel[0], start_vel[1], start_vel[2]});
    }
  }

  // optimize nominal policy
  void OptimizePolicy(int horizon, mjpc::ThreadPool& pool)  override {
    // get nominal trajectory
    this->NominalTrajectory(horizon, pool);

    // plan
    Eigen::Vector3d startPos = this->getStartPos();
    Eigen::Vector3d goalPos = this->getGoalPos();

    State<TSpace::dim> startState(startPos, this->getStartVel());

    auto starttime = std::chrono::high_resolution_clock::now();
    plan(startState, goalPos);
    auto endtime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        endtime - starttime);
    //double duration_s = double(duration.count()) / 1E6;

    std::string success = this->success() ? "Success: " : "Failure: ";
    std::cout << success << double(duration.count()) / 1000.0 << "ms"
              << std::endl;
  }

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, mjpc::ThreadPool& pool) override {}

  // set action from policy
  void ActionFromPolicy(double* action, const double* state,
                        double time, bool use_previous = false) override {
    auto policy = std::static_pointer_cast<SimpleTargetPolicy<TSpace>>(default_target_policy2_);
    Vector f = policy->alpha_ * policy->s(policy->space_.minus(this->getGoalPos(), this->getStartPos())) -
               policy->beta_ * this->getStartVel();
    action[0] = f[0]*time;
    action[1] = f[1]*time;
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
template class RMPPlanner<CylindricalSpace>;
}  // namespace rmpcpp

#endif  // RMPCPP_PLANNER_PLANNER_RMP_H
