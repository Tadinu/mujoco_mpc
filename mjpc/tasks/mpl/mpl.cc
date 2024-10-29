#include "mjpc/tasks/mpl/mpl.h"

#include <mujoco/mujoco.h>

#include <string>

#include "mjpc/utilities.h"

namespace mjpc {
std::string MPL::XmlPath() const { return GetModelPath("mpl/MPL/task.xml"); }
std::string MPL::Name() const { return "MPL"; }

// ---------- Residuals for in-hand manipulation task ---------
//   Number of residuals: 6
//     Residual (0): cube_position - palm_position
//     Residual (1): cube_orientation - cube_goal_orientation
//     Residual (2): cube linear velocity
//     Residual (3): control
//     Residual (4): hand configuration - nominal hand configuration
//     Residual (5): hand joint velocity
// ------------------------------------------------------------
void MPL::ResidualFn::Residual(const mjModel* model, const mjData* data, double* residual) const {
  int counter = 0;
  // ---------- Residual (0) ----------
  // goal position
  double* goal_position = SensorByName(model, data, "target_position");

  // system's position
  double* position = SensorByName(model, data, "obj_position");

  // position error
  mju_sub3(residual + counter, position, goal_position);
  counter += 3;

#if 1
  // ---------- Residual (1) ----------
  // Palm XY position
  double* palm_position = SensorByName(model, data, "palm_position");
  palm_position[2] = 0;

  // position error
  mju_sub3(residual + counter, position, palm_position);
  counter += 3;
#else
  // ---------- Residual (1) ----------
  // goal orientation
  double* goal_orientation = SensorByName(model, data, "target_orientation");

  // system's orientation
  double* orientation = SensorByName(model, data, "obj_orientation");
  mju_normalize4(goal_orientation);

  // orientation error
  mju_subQuat(residual + counter, goal_orientation, orientation);
  counter += 3;
#endif

#if 0
  // ---------- Residual (2) ----------
  double* obj_linear_velocity = SensorByName(model, data, "obj_linear_velocity");
  mju_copy(residual + counter, obj_linear_velocity, 3);
  counter += 3;

  // ---------- Residual (3) ----------
  mju_copy(residual + counter, data->actuator_force, model->nu);
  counter += model->nu;

  // ---------- Residual (4) ----------
  static const int qdim = 39;  // qpos size
  mju_sub(residual + counter, data->qpos + 7, model->key_qpos + 7, qdim);
  counter += qdim;

  // ---------- Residual (5) ----------
  mju_copy(residual + counter, data->qvel + 6, 17);
  counter += 17;

  // sensor dim sanity check
  CheckSensorDim(model, counter);
#endif
}

// ----- Transition for MPL task -----
//   If object is within tolerance or floor ->
//   reset its pose.
// -----------------------------------------------
void MPL::TransitionLocked(mjModel* model, mjData* data) {
#if 0
  // find object and floor
  int obj = mj_name2id(model, mjOBJ_GEOM, "object");
  int floor = mj_name2id(model, mjOBJ_GEOM, "floor");
  // look for contacts between the obj and the floor
  bool on_floor = false;
  for (int i = 0; i < data->ncon; i++) {
    mjContact* g = data->contact + i;
    if ((g->geom1 == obj && g->geom2 == floor) || (g->geom2 == obj && g->geom1 == floor)) {
      on_floor = true;
      break;
    }
  }

  double* obj_lin_vel = SensorByName(model, data, "obj_linear_velocity");
  if (on_floor && mju_norm3(obj_lin_vel) < .001) {
    // reset box pose, adding a little height
    int obj_body = mj_name2id(model, mjOBJ_BODY, "object");
    if (obj_body != -1) {
      int jnt_qposadr = model->jnt_qposadr[model->body_jntadr[obj_body]];
      int jnt_veladr = model->jnt_dofadr[model->body_jntadr[obj_body]];
      mju_copy(data->qpos + jnt_qposadr, model->qpos0 + jnt_qposadr, 7);
      mju_zero(data->qvel + jnt_veladr, 6);
    }
    mutex_.unlock();          // step calls sensor that calls Residual.
    mj_forward(model, data);  // mj_step1 would suffice, we just need contact
    mutex_.lock();
  }
#endif
}

}  // namespace mjpc
