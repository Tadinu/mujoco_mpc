#include "mjpc/tasks/mpl/mpl.h"

#include <mujoco/mujoco.h>

#include <string>

#include "mjpc/planners/cio/cio_util.h"
#include "mjpc/utilities.h"

namespace mjpc {
  std::string MPL::XmlPath() const {
    return GetModelPath(underactuated_ ? "mpl/MPL/task_underactuated.xml" : "mpl/MPL/task.xml");
  }

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
  void MPL::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
    int counter = 0;

    // Obj position
    static constexpr float OVER_BOX_OFFSET_Z = 0.01;
    double *position = SensorByName(model, data, "box");
    // position[2] += OVER_BOX_OFFSET_Z;

    // ---------- Residual (0) ----------
    // Palm XY position
    double *palm_thumb_position = SensorByName(model, data, "palm_thumb_pos");
    double *palm_pinky_position = SensorByName(model, data, "palm_pinky_pos");
    double *thumb_distal_position = SensorByName(model, data, "thumb_distal_pos");
    double *index_distal_position = SensorByName(model, data, "index_distal_pos");
    double *middle_distal_position = SensorByName(model, data, "middle_distal_pos");
    double *ring_distal_position = SensorByName(model, data, "ring_distal_pos");
    double *pinky_distal_position = SensorByName(model, data, "pinky_distal_pos");

    double palm_position[3] = {0};
    mju_add3(palm_position, palm_thumb_position, palm_pinky_position);
    mju_addTo3(palm_position, thumb_distal_position);
    mju_addTo3(palm_position, index_distal_position);
    mju_addTo3(palm_position, middle_distal_position);
    mju_addTo3(palm_position, ring_distal_position);
    mju_addTo3(palm_position, pinky_distal_position);
    mju_scl3(palm_position, palm_position, 1.0 / 7);

    // position error
    mju_sub3(residual + counter, position, palm_position);
    counter += 3;
    // std::cout << "Reach COST: " << residual[counter - 1] << std::endl;

#if 0
  // goal position
  double* goal_position = SensorByName(model, data, "target_pos");

  // position error
  mju_sub3(residual + counter, position, goal_position);
  counter += 3;
#endif

#if 1
    // ---------- Residual (1) ----------
    // position error
    residual[counter++] = cost_calc_.TotalCost();
    //std::cout << "GRASP COST: " << residual[counter - 1] << std::endl;
#else
  // ---------- Residual (1) ----------
  // goal orientation
  double* goal_orientation = SensorByName(model, data, "target_quat");

  // system's orientation
  double* orientation = SensorByName(model, data, "obj_quat");
  mju_normalize4(goal_orientation);

  // orientation error
  mju_subQuat(residual + counter, goal_orientation, orientation);
  counter += 3;
#endif

    // ---------- Residual (2) ----------
    // bring
    double *box1 = SensorByName(model, data, "box1");
    double *target1 = SensorByName(model, data, "target1");
    mju_sub3(residual + counter, box1, target1);
    counter += 3;
    double *box2 = SensorByName(model, data, "box2");
    double *target2 = SensorByName(model, data, "target2");
    mju_sub3(residual + counter, box2, target2);
    counter += 3;
#if 0
  const auto fget_hand_part_data = [&model, &data, &counter](const std::string& prefix) {
    double* position = SensorByName(model, data, (prefix + "_pos").c_str());
    // cio_utils::print_vec3(position);
    double* linear_vel = SensorByName(model, data, (prefix + "_linear_vel").c_str());
    double* angular_vel = SensorByName(model, data, (prefix + "_angular_vel").c_str());
    double* linear_acc = SensorByName(model, data, (prefix + "_linear_acc").c_str());
    double* angular_acc = SensorByName(model, data, (prefix + "_angular_acc").c_str());
  };
  fget_hand_part_data("palm_thumb");
  fget_hand_part_data("palm_pinky");
  fget_hand_part_data("thumb_distal");
  fget_hand_part_data("index_distal");
  fget_hand_part_data("middle_distal");
  fget_hand_part_data("ring_distal");
  fget_hand_part_data("pinky_distal");
#endif

#if 0
  // ---------- Residual (2) ----------
  double* obj_linear_vel = SensorByName(model, data, "obj_linear_vel");
  mju_copy(residual + counter, obj_linear_vel, 3);
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
  void MPL::TransitionLocked(mjModel *model, mjData *data) {
    // Re-setup [cost_calc_]
    residual_.cost_calc_.Setup(model, data, this);

    // Decide whether transition should happen
    double residuals[100];
    double terms[10];
    residual_.Residual(model, data, residuals);
    residual_.CostTerms(terms, residuals, /*weighted=*/false);

    // std::cout << "TERM " << weight[0] << " - " << data->userdata[0] << " - " << terms[0] << std::endl;
    //  Reach is solved:
    if (data->time > 0 && is_reaching_ && (data->ncon > 0) && terms[0] < 0.04) {
      weight[0] = 0; // disable Reach
      weight[1] = 1; // enable pick (grasp/bring)

      is_reaching_ = false;
      // std::cout << "Reach TERM " << terms[0] << std::endl;
    }

    // grasp is solved, reset:
    if (!is_reaching_ && (data->ncon == 0) && terms[1] < 0.01) {
      weight[0] = 2.5; // enable Reach
      weight[1] = 0; // disable pick (grasp/bring)
      // return stage: bring
      is_reaching_ = true;

      // std::cout << "GRASP TERM " << terms[1] << std::endl;
    }
  }
} // namespace mjpc
