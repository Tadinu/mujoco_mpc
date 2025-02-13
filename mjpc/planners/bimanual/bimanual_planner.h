#pragma once

#include <Eigen/Core>
#include <mujoco/mujoco.h>

// mjpc
#include "mjpc/core/mjpc_common.h"
#include "mjpc/planners/bimanual/cf_manager.h"
#include "mjpc/planners/bimanual/end_condition.h"
#include "mjpc/planners/bimanual/obstacle.h"
#include "mjpc/planners/bimanual/goal_type.h"
#include "mjpc/planners/planner.h"
#include "mjpc/utils/mjpc_core_util.h"
#include "mjpc/utils/mjpc_ctrl_util.h"
#include "mjpc/planners/bimanual/dual_panda_costp_controller.h"

using Eigen::Vector3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Vector8d = Eigen::Matrix<double, 8, 1>;
using Vector14d = Eigen::Matrix<double, 14, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

using ghostplanner::cfplanner::Obstacle;

// Ref: https://github.com/riddhiman13/predictive-multi-agent-framework/blob/main/src/bimanual_planning_ros/config/tasks/dual_arms_dyn1.yaml
namespace mjpc {
class PandaBimanualPlanner : public Planner {
private:
  ghostplanner::cfplanner::CfManager cf_manager_ = {};

  bool open_loop_ = true, goal_finished_ = false, switching_ = false,
       got_initial_pos_ = false;
  bool visualize_predicted_paths_ = true;
  int predicted_best_id_ = -1;
  std::vector<std::vector<Vector3d>> predicted_paths_;
  bool visualize_commanded_path_ = true;
  std::vector<Vector3d> commanded_path_;
  int8_t current_goal_id_ = -1;
  double current_dist_from_goal_ = 0;
  double time_step_ = 0.01, velocity_ = 0.2;
  Vector3d force_offset_ = {0.0, 0.0, 10.0}, last_goal_ = Vector3d::Zero(),
           obst_pos_ = Vector3d::Zero(), obst_vel_ = Vector3d::Zero(), last_pos_ = Vector3d::Zero();
  Vector14d q_goal_ = Vector14d::Zero();
  Vector6d th_min_ = {-20, -20, -20, -3, -3, -3}, th_max_ = {20, 20, 20, 3, 3, 3};
  double gesture_th_ = 10, gripper_width_ = 0.035, force_feedback_ = 0.0, force_dead_zone_ = 10.0,
         low_level_gain_ = 1.5;
  std::string type_ = "plan", robot_type_ = "dual_panda",
              message_ = "Press start planning", event_ = "moving to initial position",
              end_condition_ = "reached";
  GoalType::Type goal_type_ = GoalType::PLAN;
  bool IsPlanningActive() const { return GoalType::PLAN == goal_type_; }
  bool first_planning_run_ = true;
  std::vector<Obstacle> obstacles_ = {Obstacle("obs1", {-0.2, 0.0, 0.9}, {0.0, -0.0, 0.04}, 0.2),
                                      Obstacle("obs2", {0.2, 0.0, 0.3}, {0.0, 0.0, 0.04}, 0.225),
                                      Obstacle("obs3", {0.2, 0.0, -0.25}, {0.0, 0.0, 0.1}, 0.1),
                                      Obstacle("obs4", {100.0, 100.0, 100.0}, {0.0, 0.0, 0.0}, 0.1)
  };
  double k_attr_ = 4.0, k_circ_ = 0.015, k_repel_ = 0.08, k_damp_ = 4.0, k_manip_ = 0.0, k_repel_body_ = 0.02,
         k_goal_dist_ = 100.0,
         k_path_len_ = 10.0,
         k_safe_dist_ = 0.001, k_workspace_ = 1, gain_ = 0.0005, radius_ = 0.2, approach_dist_ = 0.25,
         detect_shell_rad_ = 0.35;
  Vector6d des_ws_limits_ = {1.0, -1.0, 0.3, -0.3, 1.1, 0.2};
  Vector7d q_goal_left_ = {-1.4219862916805648, 0.9526660423399979, 0.2692655194467807, -1.379018012046814,
                           -0.0431369121770064, 2.612734319819344, 0.36745211305883196},
           q_goal_right_ = {0.27823671439013664, 0.5812998044484216, -0.4581389926175737, -0.8199389611129149,
                            0.4614308121270604, 1.7349108616511022, -0.0180348172915559};
  size_t num_agents_ee_ = 10, num_agents_body_ = 1, max_prediction_steps_ = 1500,
         prediction_freq_multiple_ = 1;
  EndCondition ec_ = EndCondition::NONE;
  std::vector<std::string> controller_type_;
  std::vector<double> gains_, reflex_tau_max_ = {100, 100, 100, 80, 80, 40, 40},
                      reflex_F_max_ = {100, 100, 100, 30, 30, 30};

  void ErrorRecovery() {
  }

  std::chrono::steady_clock::time_point end_callback;
  void InitCFManager();

public:
  void PlanCallback(const Vector3d& p);
  void TaskCallback(GoalType goal_type = GoalType::PLAN);
  bool WaitForGesture(double F_th = 8);
  void SetThresholds(const Vector6d& F_min, const Vector6d& F_max);

  void OpenGripper(bool left, double width) {
  }

  void OpenGrippers(double width) {
  }

  void CloseGripper(bool left) {
  }

  void CloseGrippers() {
  }


  // =========================================================================================================
  // MJPC-PLANNER IMPL --
  //
  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  Task* task_ = nullptr;

  // initialize data and settings
  void Initialize(mjModel* model, const Task& task) override {
    task_ = const_cast<Task*>(&task);
    model_ = model;
    data_ = task_->data_;

    // dimensions
    dim_state_ = model->nq + model->nv + model->na; // state dimension
    dim_state_derivative_ = 2 * model->nv + model->na; // state derivative dimension
    action_dim_ = task.GetActionDim(); // action dimension
    dim_sensor_ = model->nsensordata; // number of sensor values
    dim_max_ = std::max({dim_state_, dim_state_derivative_, action_dim_, model->nuser_sensor});

    if (trajectory_) {
      trajectory_->Reset(0);
    } else {
      trajectory_ = std::make_shared<mjpc::Trajectory>();
    }

    // Init task bimanual
    if (task.IsBimanualSupported()) {
      // Init dual-arm controller
      controller_ = std::make_shared<DualPandaCoSTPController>(model, task_->data_, task_);
      controller_->init();

      // Init CFManager (starting predictive threads here-in -> To be merged to trajectory_[] later)
      current_goal_id_ = 0;
      InitCFManager();
    }
  }

  void Allocate() override {
    trajectory_->Initialize(dim_state_, action_dim_, task_->num_residual, task_->num_trace, 1);
    trajectory_->Allocate(1);
  }

  // reset memory to zeros
  void Reset(int horizon, const double* initial_repeated_action = nullptr) override {
  }

  void SetState(const State& state) override {
  }

  const Trajectory* BestTrajectory() override { return trajectory_.get(); }

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override {
#if 0
    std::vector<double> traces;
    {
      const MjpcSharedMutexLock lock(policy_mutex_);
      if (trajectory_->trace.size() >= 6) {
        traces = trajectory_->trace;
      }
    }

    static constexpr float GREEN[] = {0.0, 1.0, 0.0, 1.0};
    for (auto i = 0; (!traces.empty()) && (i < (traces.size() / 3) - 1); ++i) {
      mjpc::AddConnector(scn ? scn : task_->scene_, mjGEOM_LINE, 5,
                         (mjtNum[]){traces[3 * i], traces[3 * i + 1], traces[3 * i + 2]},
                         (mjtNum[]){traces[3 * (i + 1)], traces[3 * (i + 1) + 1], traces[3 * (i + 1) + 2]},
                         GREEN);
    }
#endif
  }

  void ClearTrace() override {
    const MjpcSharedMutexLock lock(policy_mutex_);
    trajectory_->trace.clear();
  }

  // planner-specific GUI elements
  void GUI(mjUI& ui) override {
  }

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift, int timer_shift, int planning,
             int* shift) override {
  }

  // return number of parameters optimized by planner
  int NumParameters() override { return 0; }

  // optimize nominal policy
  void OptimizePolicy(int horizon, mjpc::ThreadPool& pool) override {
    if (true) {
      const MjpcSharedMutexLock lock(policy_mutex_);
#if 1
      static constexpr int NV = 14;
      if (action_.empty()) {
        action_ = std::vector<double>(NV, 0);
      }
      static Vector14d vel = Vector14d::Zero();
      const double integration_dt = model_->opt.timestep;
      vel.head<7>() = mjpc::ControlDiff(model_, data_, "torso", "panda0_end_effector", "target",
                                        data_->qpos,
                                        integration_dt, true);
      vel.tail<7>() = mjpc::ControlDiff(model_, data_, "torso", "panda1_end_effector", "target",
                                        data_->qpos + 7,
                                        integration_dt, true);
      mju_copy(action_.data(), data_->qpos, NV);
      mju_addToScl(action_.data(), vel.data(), 1, NV);
      //print(action_);
#else
      TaskCallback(GoalType::PLAN);
      const auto tau_d = controller_->update();
      action_ = std::vector<double>(tau_d.data(), tau_d.data() + tau_d.size());
#endif
    }
  }

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, mjpc::ThreadPool& pool) override {
  }

  // set action from policy
  void ActionFromPolicy(double* action, const double* state, double time, bool use_previous) override {
    const MjpcSharedMutexLock lock(policy_mutex_);

    // WAIT FOR ACTION TO BE COMPUTED
    if (action_.empty()) {
      return;
    }

    // APPLY ACTION: COPY [action_] -> [action]
#if 0
    const mjtNum* target_pos = task_->QueryTargetPos();
    if (target_pos) {
      trajectory_->trace.push_back(target_pos[0]);
      trajectory_->trace.push_back(target_pos[1]);
      trajectory_->trace.push_back(target_pos[2]);
    }
#endif
    MJPC_PRINTDB("ACTION", action_);
    mju_copy(action, action_.data(), action_.size());
    // Clear [action_]
    action_.clear();

    // Clamp controls on outputted [action]
    mjpc::Clamp(action, model_->actuator_ctrlrange, model_->nu);
  }

protected:
  // mjpc
  std::shared_ptr<mjpc::Trajectory> trajectory_ = nullptr;
  int dim_state_ = 0; // state
  int dim_state_derivative_ = 0; // state derivative
  int dim_sensor_ = 0; // output (i.e., all sensors)
  int dim_max_ = 0; // maximum dimension
  mutable std::shared_mutex policy_mutex_;
  // [action_] is shared among policy motion planning threads.
  std::vector<double> action_;
  DualPandaCoSTPControllerPtr controller_ = nullptr;
};
} // namespace mjpc
