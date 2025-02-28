#include "mjpc/tasks/lsqp/lsqp.h"

#include <string>

// mujoco
#include <mujoco/mujoco.h>

// mjpc
#include "mjpc/utilities.h"
#include "mjpc/planners/lsqp/lsqp_planner.h"

namespace mjpc {
const std::vector<std::string> Lsqp::FINGERTIP_NAMES = {"rf_tip", "mf_tip", "ff_tip", "th_tip"};
const std::map<std::string, std::vector<float>> Lsqp::FINGERTIPS_RGBA = {
    {FINGERTIP_NAMES[0], {0.9f, 0.f, 0.f, 1.f}}, // Red
    {FINGERTIP_NAMES[1], {0.f, 0.9f, 0.f, 1.f}}, // Green
    {FINGERTIP_NAMES[2], {0.f, 0.f, 0.9f, 1.f}}, // Blue
    {FINGERTIP_NAMES[3], {0.9f, 0.9f, 0.9f, 1.f}}}; // White

void Lsqp::SetPlanner(Planner* planner) {
  Task::SetPlanner(planner);
  lsqp_planner_ = dynamic_cast<LsqpPlanner*>(planner);
}

void Lsqp::TransitionLocked(mjModel* model, mjData* data) {
  Task::TransitionLocked(model, data);
  double residuals[100];
  double terms[10];
  residual_.Residual(model, data, residuals);
  residual_.CostTerms(terms, residuals, /*weighted=*/false);
  //mjpc::print("WEIGHT:", weight[0], weight[1]);
  //mjpc::print("TERMS:", terms[0], terms[1]);

  // reach is solved:
  auto& norm_type = data->userdata[0];
  const auto& reach_distance = terms[0]; // Distance to target object
  if (data->time > 0 && norm_type == 0 && reach_distance < 0.04) {
    weight[0] = 0; // disable reach
    weight[1] = 1; // enable bring
    norm_type = 2;
  }

  // bring is solved, reset:
  const auto& bring_distance = terms[1]; // Distance to target goal
  if (norm_type == 2 && bring_distance < 0.01) {
    weight[0] = 1; // enable reach
    weight[1] = 0; // disable bring
    norm_type = 0;
  }

  // Init once is already checked here-in
  InitMocaps();

  // Reset target obj if being flung away
  if (mju_dist3(mjpc::QueryBodyPos(model_, data, Lsqp::TARGET_OBJ_NAME), (double[3]){0, 0, 0}) > 2) {
    int obj_id = mj_name2id(model, mjOBJ_BODY, TARGET_OBJ_NAME);
    if (obj_id != -1) {
      int jnt_qposadr = model->jnt_qposadr[model->body_jntadr[obj_id]];
      int jnt_veladr = model->jnt_dofadr[model->body_jntadr[obj_id]];
      mju_copy(data->qpos + jnt_qposadr, model->qpos0 + jnt_qposadr, 7);
      mju_zero(data->qvel + jnt_veladr, 6);
    }
    mutex_.unlock(); // step calls sensor that calls Residual.
    mj_forward(model, data); // mj_step1 would suffice, we just need contact
    mutex_.lock();
  }
}

void Lsqp::ResidualFn::Residual(const mjModel* model, const mjData* data, double* residual) const {
  const Lsqp* lsqp_task = static_cast<const Lsqp*>(task_);
  int counter = 0;

  // Obj position, quat
  double* obj_pos = SensorByName(model, data, std::string(TARGET_OBJ_NAME) + "_pos");
  double* obj_quat = SensorByName(model, data, std::string(TARGET_OBJ_NAME) + "_quat");

  // ---------- Residual (0) ----------
  // EE target position
  double* ee_target_pos = SensorByName(model, data, lsqp_task->EETargetSiteName() + "_pos");

  // position error
  mju_sub3(residual + counter, obj_pos, ee_target_pos);
  counter += 3;

  // ---------- Residual (1) ----------
  // goal position error
  mju_sub3(residual + counter, mjpc::QuerySitePos(model, data, TARGET_OBJ_GOAL_NAME), obj_pos);
  counter += 3;

  // goal orientation error
  mju_subQuat(residual + counter, mjpc::QuerySiteQuat(model, data, TARGET_OBJ_GOAL_NAME), obj_quat);
  counter += 4;

#if 0
  // ---------- Residual (2) ----------
  // grasp error
  residual[counter++] = cost_calc_.TotalCost();
  //std::cout << "GRASP COST: " << residual[counter - 1] << std::endl;
#endif

  // Sanity check
  CheckSensorDim(model, counter);
}
} // namespace mjpc
