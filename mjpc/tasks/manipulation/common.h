// Copyright 2023 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// a library of common cost terms and utilities used in manipulation tasks in
// this directory

#ifndef MJPC_MJPC_TASKS_MANIPULATION_COMMON_H_
#define MJPC_MJPC_TASKS_MANIPULATION_COMMON_H_

#include <vector>

#include <mujoco/mujoco.h>

namespace mjpc::manipulation {

// visual groups used in the XML model
enum VisualGroup {
  kGroupVisualScene = 0,     // meshes for static scene and movable objects
  kGroupVisualRobot = 1,     // visual meshes for parts of the robot
  kGroupCollisionScene = 3,  // collision geometry for scene and objects
  kGroupCollisionRobot = 4,  // collision geometry for parts of the robot
  kGroupSites = 5,           // mocap bodies for debugging
};

// various IDs and values extracted from the XML model at Reset time
struct ModelValues {
  static ModelValues FromModel(const mjModel* model) {
    ModelValues values;
    values.robot_body_id = mj_name2id(model, mjOBJ_BODY, "link0");
    values.gripper_site_id = mj_name2id(model, mjOBJ_SITE, "pinch");

    for (const char* name :
        {"right_pad1", "right_pad2", "left_pad1", "left_pad2"}) {
      values.gripper_pad_geoms.push_back(mj_name2id(model, mjOBJ_GEOM, name));
    }
    values.left_finger_pad_geom_id =
        mj_name2id(model, mjOBJ_GEOM, "left_pad2");
    values.right_finger_pad_geom_id =
        mj_name2id(model, mjOBJ_GEOM, "right_pad2");

    values.gripper_actuator = mj_name2id(model, mjOBJ_ACTUATOR, "fingers");
    values.open_gripper_ctrl =
        model->actuator_ctrlrange[2 * values.gripper_actuator];
    values.closed_gripper_ctrl =
        model->actuator_ctrlrange[2 * values.gripper_actuator + 1];
    values.gripper_ctrl_range =
        values.closed_gripper_ctrl - values.open_gripper_ctrl;

    int target_mocap_body_id = mj_name2id(model, mjOBJ_BODY, "target_mocap");
    if (target_mocap_body_id != -1) {
      // this mocap body isn't present in all manipulation models
      values.target_mocap_body = model->body_mocapid[target_mocap_body_id];
    }
    return values;
  }


  int robot_body_id = -1;
  int gripper_site_id = -1;
  std::vector<int> gripper_pad_geoms;
  int left_finger_pad_geom_id = -1;
  int right_finger_pad_geom_id = -1;
  int gripper_actuator = -1;

  // ctrl values for open and closed gripper
  double open_gripper_ctrl = 0;
  double closed_gripper_ctrl = 0;
  double gripper_ctrl_range = 0;

  // mocapid for a body that shows where the target is
  int target_mocap_body = -1;
};

// computes a control cost and writes it to residual, returns the number of
// elements written (model->nu).
int ComputeControlCost(const mjModel* model, const mjData* data,
                       double* residual);

// computes a cost on the relative velocity between the gripper and an object,
// and writes it to residual, returns the number of elements written (3).
int ComputeGripperVelocityCost(const mjModel* model, const mjData* data,
                               const ModelValues& model_vals,
                               const double* object_linvel, double* residual);

// computes a cost on the maximum velocity of any moving object in the scene,
// given a vector of relevant object body IDs. returns the number of elements
// written to residual (6).
int ComputeMaxVelocityCost(const mjModel* model, const mjData* data,
                           const std::vector<int>& body_ids, double* residual);

// returns 1 if there's a body other than exclude_body_id between pos and
// object, ignoring any parts of the robot model
double BlockedCost(const mjModel* model, const mjData* data,
                   const ModelValues& model_vals, const double* pos,
                   const double* object, int exclude_body_id);

double GraspQualityCost(const mjModel* m, const mjData* d,
                        const ModelValues& model_vals, int body_id,
                        bool release);

// returns a cost term that is high if there are large forces between the robot
// and bodies that aren't object_body_id
inline double CarefulCost(const mjModel* model, const mjData* data,
                          const ModelValues& model_vals, int object_body_id) {
  double result = 0;
  for (int i = 0; i < data->ncon; i++) {
    int b1 = model->geom_bodyid[data->contact[i].geom1];
    int b2 = model->geom_bodyid[data->contact[i].geom2];
    int r1 = model->body_rootid[b1];
    int r2 = model->body_rootid[b2];
    if (r1 == model_vals.robot_body_id || r2 == model_vals.robot_body_id) {
      // contact with robot
      if (r2 == object_body_id || r1 == object_body_id) {
        continue;  // contact with the object is okay
      }
      mjtNum force[6];
      mj_contactForce(model, data, i, force);
      result += mju_norm3(force);
    }
  }
  return mju_log10(result + 1);
}

// finds a position between the gripper pads and writes it to pos
inline void ComputeRobotiqHandPos(const mjModel* model, const mjData* data,
                                  const ModelValues& model_vals, double* pos) {
  for (int geom_id : model_vals.gripper_pad_geoms) {
    mju_addToScl3(pos, data->geom_xpos + 3 * geom_id, 0.25);
  }
}
}  // namespace mjpc::manipulation

#endif  // MJPC_MJPC_TASKS_MANIPULATION_COMMON_H_
