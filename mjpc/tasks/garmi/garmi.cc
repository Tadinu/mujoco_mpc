#include "mjpc/tasks/garmi/garmi.h"

#include <cassert>
#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

// mjpc
#include "mjpc/planners/lsqp/lsqp_config.h"
#include "mjpc/planners/lsqp/lsqp_solver.h"

namespace mjpc {
using absl::Uniform;

void Garmi::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                 double* residual) const {
  int counter = 0;

  // reach left, encourage proper alignment
  double* box_to_left_gripper = SensorByName(model, data, "box_to_left_gripper");
  mju_copy3(residual + counter, box_to_left_gripper);
  // The sensor is "object pos in the frame of the gripper".
  // X points forward in the gripper frame.
  // By making the distance larger in Y and Z, the gripper is encouraged to
  // orient itself towards the object.
  residual[counter + 1] *= 2;
  residual[counter + 2] *= 2;
  counter += 3;

  // reach right, encourage proper alignment
  double* box_to_right_gripper = SensorByName(model, data, "box_to_right_gripper");
  mju_copy3(residual + counter, box_to_right_gripper);
  residual[counter + 1] *= 2;
  residual[counter + 2] *= 2;
  counter += 3;

  // grasp

  // normal arrays, counters
  double normal[4][3] = {{0}, {0}, {0}, {0}};
  int nnormal[4] = {0, 0, 0, 0};

  // get body ids, object position
  int finger[4] = {0};
  char name[] = "finger__";
  char finger_name[4][3] = {"LL", "LR", "RL", "RR"};
  for (int segment = 0; segment < 4; segment++) {
    name[6] = finger_name[segment][0];
    name[7] = finger_name[segment][1];
    int finger_sensor_id = mj_name2id(model, mjOBJ_SENSOR, name);
    finger[segment] = model->sensor_objid[finger_sensor_id];
    assert(finger[segment] > 0);
  }
  int target_sensor_id = mj_name2id(model, mjOBJ_SENSOR, "box");
  int object_id = model->sensor_objid[target_sensor_id];
  assert(object_id > 0);

  // loop over contacts, add up (and maybe flip) relevant normals
  int ncon = data->ncon;
  for (int i = 0; i < ncon; ++i) {
    const mjContact* con = data->contact + i;
    int body[2] = {model->geom_bodyid[con->geom[0]],
                   model->geom_bodyid[con->geom[1]]};
    for (int j = 0; j < 2; ++j) {
      if (body[j] == object_id) {
        for (int k = 0; k < 4; ++k) {
          if (body[1 - j] == finger[k]) {
            // We want the normal to point from the finger to the object.
            // In mjContact the normal always points from body[0] to body[1].
            // Since body[j] is the object, if j == 0, the normal is flipped.
            double sign = (j == 0) ? -1 : 1;
            mju_addToScl3(normal[k], con->frame, sign);
            nnormal[k]++;
          }
        }
      }
    }
  }

  // grasp residual
  double grasp = 1;

  // left hand
  if (nnormal[0] && nnormal[1]) {
    mju_normalize3(normal[0]);
    mju_normalize3(normal[1]);
    // we want the two normal directions to be opposite each other
    grasp = 0.5 * (mju_dot3(normal[0], normal[1]) + 1);
  }
  residual[counter] = grasp;

  // multiply by right hand
  // the multiplication results in a term that means
  // "one of the hands should grasp the object well"
  if (nnormal[2] && nnormal[3]) {
    mju_normalize3(normal[2]);
    mju_normalize3(normal[3]);
    grasp = 0.5 * (mju_dot3(normal[2], normal[3]) + 1);
    residual[counter] *= grasp;
  }

  // take geometric mean
  residual[counter] = mju_sqrt(mju_max(0, residual[counter]));

  counter++;

  // bring
  double* target = SensorByName(model, data, "target");
  double* box = SensorByName(model, data, "box");
  mju_sub3(residual + counter, box, target);
  counter += 3;

  CheckSensorDim(model, counter);
}

void Garmi::TransitionLocked(mjModel* model, mjData* data) {
  Task::TransitionLocked(model, data);
  double* box = SensorByName(model, data, "box");
  double* target = SensorByName(model, data, "target");
  double vec[3];
  mju_sub3(vec, box, target);
  double dist = mju_norm3(vec);

  // in case user manually reset the env
  if (data->time < last_solve_time_) {
    last_solve_time_ = data->time;
  }

  int target_id = mj_name2id(model, mjOBJ_GEOM, "target");
  assert(target_id > 0);

  // reset target on success
  if (data->time > 0 && dist < model->geom_size[target_id * 3]) {
    absl::BitGen gen_;

    // move target
    double flip = target[0] > 0 ? -1 : 1;
    data->mocap_pos[0] = flip * Uniform<double>(gen_, .3, .4);
    double side = Uniform<double>(gen_, 0, 1) > 0.5 ? -1 : 1;
    data->mocap_pos[1] = side * Uniform<double>(gen_, .2, .3);
    data->mocap_pos[2] = Uniform<double>(gen_, 0.25, 0.7);

    // set solve time
    last_solve_time_ = data->time;
  }

  int nq = model->nq;
  int nv = model->nv;

  // reset box if it falls off table
  if (box[2] < -0.1) {
    // assumes that free body's freejoint is the last joint
    // and that 'home' is the first keyframe
    mju_copy3(data->qpos + nq - 7, model->key_qpos + nq - 7);
    mju_zero3(data->qvel + nv - 6);
  }

  // reset arms if no solution after 30 seconds
  constexpr int kMaxSolveTime = 30;
  if (data->time > last_solve_time_ + kMaxSolveTime) {
    mju_copy(data->qpos, model->key_qpos, nq);

    // set solve time
    last_solve_time_ = data->time;
  }
}

void Garmi::InitSolverConfigs(const LsqpSolverPtr& solver, const mjData* data, int ndofs) {
  // 1- Create config
  solver->config_ = LsqpConfig(model_, ndofs);

  // 2- Tasks
  // 2.1- End-effector task
  solver->end_effector_subtasks_.emplace_back(LsqpRelativeFrameTask("Left_EE", model_,
                                                                    "left_ee_site",
                                                                    mjOBJ_SITE,
                                                                    "left_base",
                                                                    mjOBJ_SITE,
                                                                    /*position_cost*/
                                                                    Eigen::VectorXd::Constant(1, 5.0),
                                                                    /*orientation_cost*/
                                                                    Eigen::VectorXd::Constant(1, 1.0),
                                                                    /*gain*/ 1.0,
                                                                    /*lm_damping*/1.0));
  solver->end_effector_subtasks_.emplace_back(LsqpRelativeFrameTask("Right_EE", model_,
                                                                    "right_ee_site",
                                                                    mjOBJ_SITE,
                                                                    "right_base",
                                                                    mjOBJ_SITE,
                                                                    /*position_cost*/
                                                                    Eigen::VectorXd::Constant(1, 5.0),
                                                                    /*orientation_cost*/
                                                                    Eigen::VectorXd::Constant(1, 1.0),
                                                                    /*gain*/ 1.0,
                                                                    /*lm_damping*/1.0));
  for (auto& ee_task : solver->end_effector_subtasks_) {
    solver->subtasks_.push_back(&ee_task);
  }

  // 2.2- Posture task (as biased pose in diff-ik solving, with cost being smaller than other tasks)
  if (!system_qpos_home_.empty()) {
    solver->posture_subtask_ = LsqpPostureTask("Posture", model_, /*cost*/
                                               Eigen::VectorXd::Constant(1, 1e-3));
    solver->posture_subtask_.SetTarget(PosToEigen(system_qpos_home_.data(), ndofs));
    solver->subtasks_.push_back(&solver->posture_subtask_);
  }

#if 0
  // 2.3- Damping task
  Eigen::VectorXd immobile_base_cost = 100.0 * Eigen::VectorXd::Ones(ndofs);
  immobile_base_cost[2] = 1e-3;
  solver->damping_subtask_ = LsqpDampingTask("Damping", model_, immobile_base_cost);
  solver->subtasks_.push_back(&solver->damping_subtask_);
#endif

  // 3- Config limits
  solver->config_limits_ = {std::make_shared<LsqpPositionLimit>(model_, ndofs)};
}

std::vector<double> Garmi::Control(double* policy_action, mjData* data,
                                   const LsqpSolverPtr& solver) {
  // Use task data if not running in worker thread in MPC rollouts
  const bool is_rollout_thread_data = (nullptr != data);
  if (!is_rollout_thread_data) {
    data = data_;
    if (!data_) {
      return {};
    }
  }

  // [Lsqp solver]: solve diff-ik
  const auto& cur_solver = is_rollout_thread_data ? solver : lsqp_solver_;
  const std::vector<double> ctrl = Solve(cur_solver, data);
  return mjpc::Lsqp::POSITION_CTRL_ENABLED
           ? (cur_solver->Config().CheckJointValues(
                  ctrl.data(), mjpc::GARMI_ACTUATED_JOINT_NAMES, 0.1)
                ? ctrl
                : mjpc::InvalidControls(ctrl.size()))
           : ctrl;
}

std::vector<double> Garmi::Solve(const LsqpSolverPtr& solver, const mjData* data) {
  // Update [end-effector task]'s mocap target
  const auto config = solver->Config();
  solver->end_effector_subtasks_[0].SetTarget(config.GetTransform(data, "left_target", mjOBJ_BODY,
                                                                  "left_base", mjOBJ_SITE));
  solver->end_effector_subtasks_[1].SetTarget(config.GetTransform(data, "right_target", mjOBJ_BODY,
                                                                  "right_base", mjOBJ_SITE));

  // Compute velocity and integrate into the next configuration
  const Eigen::VectorXd vel = mjpc::IK_Solve(data, config, solver->subtasks_,
                                             INTEGRATION_DT, /*damping*/1e-3,
                                             solver->ConfigLimits());
  // NOTE: vel.size() == model->nv (qvel's size == num of dofs)
  std::vector<double> ctrl;
  mjpc::print("SOLVED VEL", std::vector<double>(vel.data(), vel.data() + vel.size()));
  if (!vel.hasNaN()) {
    if constexpr (Lsqp::POSITION_CTRL_ENABLED) {
      // Integrate [vel] into current [q]
      // Init [ctrl] as current [data->qpos] -> NOTE: Must always get the full qpos[nq]
      ctrl = std::vector<double>(data->qpos, data->qpos + model_->nq);
      mj_integratePos(model_, ctrl.data(), vel.data(), INTEGRATION_DT);
    } else {
      // [vel] -> [ctrl]
      ctrl = std::vector<double>(vel.size(), 0.0);
      mju_copy(ctrl.data(), vel.data(), vel.size());
    }
  } else {
    mju_zero(ctrl.data(), ctrl.size());
  }

  return ctrl;
}
} // namespace mjpc
