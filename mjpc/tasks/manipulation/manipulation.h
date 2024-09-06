// Copyright 2022 DeepMind Technologies Limited
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

#ifndef MJPC_MJPC_TASKS_MANIPULATION_MANIPULATION_H_
#define MJPC_MJPC_TASKS_MANIPULATION_MANIPULATION_H_

#include <mujoco/mujoco.h>

#include <Eigen/Eigen>

#include "mjpc/ddg/include/ddg_body.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/rmp/include/util/rmp_util.h"
#include "mjpc/task.h"
#include "mjpc/tasks/manipulation/common.h"
#include "mjpc/utilities.h"

namespace mjpc::manipulation {
class Bring : public Task {
public:
  Bring() : residual_(this, ModelValues()) { actuator_kv = 1.f; }
  std::string Name() const override;
  std::string XmlPath() const override;
  std::string URDFPath() const override;
  std::string GetBaseBodyName() const override {
    static std::string name = "panda_link0";
    return name;
  }
  std::vector<std::string> GetEndtipNames() const override {
    // Ones in URDF, not XML
    static std::vector<std::string> names = {"panda_leftfinger", "panda_rightfinger"};
    return names;
  }
  std::vector<std::string> GetCollisionLinkNames() const override {
    // Ones in URDF, not XML
    // NOTE: In XML, link5 collisions, composed of 3 subparts, is not obvious to fetch
    static std::vector<std::string> names = {"panda_hand", "panda_link3", "panda_link4"};
    return names;
  }
  FabLinkCollisionProps GetCollisionLinkProps() const override {
    const auto& link_names = GetCollisionLinkNames();
    static FabLinkCollisionProps props = {
        {link_names[0], {0.02}}, {link_names[1], {0.02}}, {link_names[2], {0.02}}};
    return props;
  }
  FabSelfCollisionNamePairs GetSelfCollisionNamePairs() const override {
    // Ones in URDF, not XML
    return {{"panda_hand", {"panda_link2", "panda_link4"}}};
  }
  int GetStaticObstaclesNum() const override { return 3; }
  int GetDynamicObstaclesNum() const override {
    return (planner_ && planner_->tuning_on_) ? static_cast<int>(GetCollisionLinkNames().size()) : 0;
  }
  int GetPlaneConstraintsNum() const override { return 1; }

  int GetActionDim() const override { return (GetSubGoals()[0]->child_link_name() == "panda_hand") ? 7 : 9; }
  std::vector<FabSubGoalPtr> GetSubGoals() const override {
    // Static subgoals with static [desired_position]
    static std::vector<FabSubGoalPtr> subgoals = {std::make_shared<FabStaticSubGoal>(FabSubGoalConfig{
        .name = "subgoal0",
        .type = FabSubGoalType::STATIC,
        .is_primary_goal = true,
        .epsilon = 0.05,
        .indices = {0, 1, 2},
        .weight = 0.7,
        .parent_link_name = "panda_link0",
        .child_link_name = "panda_hand",
    })};
#if 0
        std::make_shared<FabStaticSubGoal>(
            FabSubGoalConfig{.name = "subgoal1",
                             .type = FabSubGoalType::STATIC,
                             .is_primary_goal = false,
                             .epsilon = 0.05,
                             .indices = {0, 1, 2},
                             .weight = 6.0,
                             .parent_link_name = "panda_link7",
                             .child_link_name = "panda_hand",
                             .desired_position = {6.55186038e-18, 0.00000000e+00, -1.07000000e-01}}),

        std::make_shared<FabStaticJointSpaceSubGoal>(
            FabSubGoalConfig{.name = "subgoal2",
                             .type = FabSubGoalType::STATIC_JOINT_SPACE,
                             .is_primary_goal = false,
                             .epsilon = 0.05,
                             .indices = {6},
                             .weight = 6.0,
                             .parent_link_name = "panda_link0",
                             .child_link_name = "panda_hand",
                             .desired_position = {M_PI_4}})};

    static const auto link0_id = QueryBodyId("link0");  // GetBaseBodyName() - "pand_"
    static const mjtNum* link0_pos = &data_->xpos[3 * link0_id];
    FAB_PRINTDB(link0_id, link0_pos[0], link0_pos[1], link0_pos[2]);
    mju_subFrom3(subgoals[0]->cfg_.desired_position.data(), link0_pos);
    FAB_PRINTDB("Subgoal0 pos", subgoals[0]->cfg_.desired_position);
#endif

    const auto* goal_pos = GetGoalPos();
    mju_copy3(subgoals[0]->cfg_.desired_position.data(), goal_pos);
    if (is_static) {
      return subgoals;
    } else {
      subgoals[0]->cfg_.type = FabSubGoalType::DYNAMIC;
      const auto* goal_vel = GetGoalVel();
      mju_copy3(subgoals[0]->cfg_.desired_vel.data(), goal_vel);

      const auto* goal_acc = GetGoalAcc();
      mju_copy3(subgoals[0]->cfg_.desired_acc.data(), goal_acc);
      FAB_PRINT("Goal pos", goal_pos[0], goal_pos[1], goal_pos[2]);
      FAB_PRINT("Goal vel", goal_vel[0], goal_vel[1], goal_vel[2]);
      FAB_PRINT("Goal acc", goal_acc[0], goal_acc[1], goal_acc[2]);
      return {subgoals[0]};
    }
  }

  bool AreObstaclesFixed() const override { return true; }
  int GetDynamicObstaclesDimension() const override { return 3; }
  std::vector<FabJointLimit> GetJointLimits() const override {
    return {{-2.8973, 2.8973},   // panda_joint1
            {-1.7628, 1.7628},   // panda_joint2
            {-2.8973, 2.8973},   // panda_joint3
            {-3.0718, -0.0698},  // panda_joint4
            {-2.8973, 2.8973},   // panda_joint5
            {-0.0175, 3.7525},   // panda_joint6
            {-2.8973, 2.8973}};  // panda_joint7
  }

  // NOTES on mutex:
  // Access to model & data: already locked by [sim.mtx]
  // Access to task local data: lock on [task_data_mutex_]
  int GetTargetObjectId() const override { return QueryBodyId("target"); }
  int GetTargetObjectGeomId() const override { return QueryGeomId("target/target_00_collision"); }

  void SetGoalPos(const double* pos) const { SetBodyMocapPos("target", pos); }
  const mjtNum* GetGoalPos() const override {
    return IsGoalFixed() ? QueryTargetPos() : QueryBodyMocapPos("target");
  }
  const mjtNum* GetGoalVel() const override { return QueryTargetVel(); }
  const mjtNum* GetGoalAcc() const override { return QueryTargetAcc(); }
  bool IsGoalFixed() const override { return is_static; }
  virtual void MoveObstacles();

  bool QueryGoalReached() override { return false; }
  void QueryObstacleStatesX() override {
    MJPC_LOCK_TASK_DATA_ACCESS;
    obstacle_statesX_.clear();
    const auto fQueryObstacle = [this](const std::string& obst_xml_name,
                                       const std::string& obst_collision_name) {
      const auto obstacle_i_id = QueryBodyId(obst_xml_name.c_str());
      const auto obstacle_geom_i_id = QueryGeomId(obst_collision_name.c_str());
      assert(obstacle_i_id >= 0);
      mjtNum* obstacle_i_size =
          (obstacle_geom_i_id >= 0) ? &model_->geom_size[3 * obstacle_geom_i_id] : nullptr;
      mjtNum* obstacle_i_pos = &data_->xpos[3 * obstacle_i_id];
      mjtNum* obstacle_i_rot = &data_->xquat[4 * obstacle_i_id];

      static constexpr int LIN_IDX = 3;
#if 1
      mjtNum obstacle_i_full_vel[6];  // rot+lin
      mj_objectVelocity(model_, data_, mjOBJ_BODY, obstacle_i_id, obstacle_i_full_vel,
                        /*flg_local=*/0);
      mjtNum obstacle_i_lin_vel[StateX::dim];
      memcpy(obstacle_i_lin_vel, &obstacle_i_full_vel[LIN_IDX], sizeof(mjtNum) * StateX::dim);

      mjtNum obstacle_i_full_acc[6];  // rot+lin
      mj_objectAcceleration(model_, data_, mjOBJ_BODY, obstacle_i_id, obstacle_i_full_acc,
                            /*flg_local=*/0);
      mjtNum obstacle_i_lin_acc[StateX::dim];
      memcpy(obstacle_i_lin_acc, &obstacle_i_full_acc[LIN_IDX], sizeof(mjtNum) * StateX::dim);
#else
      const auto obstacle_i_lin_idx = 6 * obstacle_i_id + LIN_IDX;
      mjtNum obstacle_i_lin_vel[StateX::dim];
      mju_copy(obstacle_i_lin_vel, &data_->cvel[obstacle_i_lin_idx], StateX::dim);

      mjtNum obstacle_i_lin_acc[StateX::dim];
      mju_copy(obstacle_i_lin_acc, &data_->cacc[obstacle_i_lin_idx], StateX::dim);
#endif
      obstacle_statesX_.push_back(StateX{
          .pos_ = rmp::vectorFromScalarArray<StateX::dim>(obstacle_i_pos),
          .rot_ = rmp::quatFromScalarArray<StateX::dim>(obstacle_i_rot).toRotationMatrix(),
          .vel_ = rmp::vectorFromScalarArray<StateX::dim>(obstacle_i_lin_vel),
          .acc_ = rmp::vectorFromScalarArray<StateX::dim>(obstacle_i_lin_acc),
          .size_ =
              rmp::vectorFromScalarArray<StateX::dim>(obstacle_i_size ? obstacle_i_size : (mjtNum[]){})});
    };

    // Env obstacles
    fQueryObstacle("obstacle_0", "obstacle_0");
    fQueryObstacle("obstacle_1", "obstacle_1");

    // Body arm links as obstacles
    // NOTE: As observed, unclear why yet involving body links (as obstacles) disrupt the arm ik planning
    if (planner_ && planner_->tuning_on_) {
      static const int prefix_len = std::string("panda_").size();
      for (const auto& link_urdf_name : GetCollisionLinkNames()) {
        const auto link_xml_name = link_urdf_name.substr(prefix_len);
        // NOTE:
        // WIP-["link5"]: This requires GetDynamicObstaclesNum() to be updated to match the total no of
        // collision links + free obsts
        if (link_urdf_name == "link5") {
          fQueryObstacle("link5", "link5_c0");
          fQueryObstacle("link5", "link5_c1");
          fQueryObstacle("link5", "link5_c2");
        } else {
          fQueryObstacle(link_xml_name, link_xml_name + "_c");
        }
      }
    }
  }

  std::vector<double> QueryJointPos(int dof) const override {
    if (model_ && data_) {
      std::vector<double> qpos(dof, 0);
      memcpy(qpos.data(), data_->qpos + model_->jnt_qposadr[mj_name2id(model_, mjOBJ_JOINT, "joint1")],
             std::min(model_->nq, dof) * sizeof(double));
      return qpos;
    }
    return {};
  }

  std::vector<double> QueryJointVel(int dof) const override {
    if (model_ && data_) {
      std::vector<double> qvel(dof, 0);
      memcpy(qvel.data(), data_->qvel + model_->jnt_dofadr[mj_name2id(model_, mjOBJ_JOINT, "joint1")],
             std::min(model_->nv, dof) * sizeof(double));
      return qvel;
    }
    return {};
  }

  class ResidualFn : public mjpc::BaseResidualFn {
  public:
    explicit ResidualFn(const Bring* task, ModelValues values)
        : mjpc::BaseResidualFn(task), model_vals_(std::move(values)) {}

    void Residual(const mjModel* model, const mjData* data, double* residual) const override;

  private:
    friend class Bring;
    ModelValues model_vals_;
  };

  void Initialize() override {
    mug_ = std::make_shared<DDGBody>();

#if 0
    // TODO: mj-recompile not working yet
    static bool dynamic_obj_required = true;
    if (dynamic_obj_required) {
      static constexpr float BLUE[] = {0.0, 0.0, 1.0, 1.0};
      static const std::string mesh_name = "mug";
      mjpc::AddBody(model_, data_, mjGEOM_MESH, (double[]){1.0}, (double[]){0, 0, 5}, BLUE,
                    (mjString*)&mesh_name);
      dynamic_obj_required = false;
    }
#endif
  }
  void TransitionLocked(mjModel* model, mjData* data) override;
  void ResetLocked(const mjModel* model) override;

  void ModifyScene(const mjModel* model, const mjData* data, mjvScene* scene) const override {
    if (is_static) {
      // Draw goal
      static constexpr float GREEN[] = {0.0, 1.0, 0.0, 1.0};
      mjpc::AddGeom(scene, mjGEOM_SPHERE, (mjtNum[]){0.02}, GetGoalPos(), /*mat=*/nullptr, GREEN);
    }

    // Draw pinch
    static constexpr float BLUE[] = {0.0, 0.0, 1.0, 1.0};
    double pinch_pos[3];
    mju_copy(pinch_pos, &data_->site_xpos[3 * mj_name2id(model, mjOBJ_SITE, "pinch")], 3);
    mjpc::AddGeom(scene, mjGEOM_SPHERE, (mjtNum[]){0.02}, pinch_pos, /*mat=*/nullptr, BLUE);

    // DDG Body
    const auto& mug_geom = mug_->geometry;
    const auto mug_id = mj_name2id(model_, mjOBJ_BODY, "mug");
    mjtNum mug_pos[3];
    mju_copy3(mug_pos, &data_->xipos[3 * mug_id]);
    mjtNum mug_quat[4];
    mju_copy4(mug_quat, &data_->xquat[4 * mug_id]);

#if 0
    std::cout << mug_pos[0] << "-" << mug_pos[1] << "-" << mug_pos[2] << "-" << mug_quat[0] << "-"
              << mug_quat[1] << "-" << mug_quat[2] << "-" << mug_quat[3] << std::endl;
#endif
    mug_->transform(mug_pos, mug_quat, 0.01);

#if 0
    static bool t = false;
    for (auto i = 0; i < mug_->mesh->nVertices(); ++i) {
      const auto& vpos = mug_geom->inputVertexPositions[i];
      const auto& vcolor = mug_->K_colors[i];
      mjpc::AddGeom(scene, mjGEOM_SPHERE, (mjtNum[]){0.001}, (mjtNum[]){vpos[0], vpos[1], vpos[2]},
                    /*mat=*/nullptr, (float[]){float(vcolor[0]), float(vcolor[1]), float(vcolor[2]), 1.0});
    }
#else
    for (const auto& v : mug_->mesh->vertices()) {
      // const auto ew = mug_geom->vertexNormalEquallyWeighted(v);
      // const auto tw = mug_geom->vertexNormalAngleWeighted(v);
      // const auto si = mug_geom->vertexNormalSphereInscribed(v);
      // const auto an = mug_geom->vertexNormalAreaWeighted(v);
      // const auto hn = mug_geom->vertexNormalMeanCurvature(v);
      const auto kn = mug_geom->vertexNormalGaussianCurvature(v);

      const auto pos = mug_geom->inputVertexPositions[v];
      const Eigen::Vector3<double> curPos = {pos.x, pos.y, pos.z};
      const Eigen::Vector3<double> curVel = {kn.x, kn.y, kn.z};
      const Eigen::Vector3<double> curVelEnd = curPos + curVel * 0.1;
      mjpc::AddConnector(scene, mjGEOM_ARROW, 0.001, curPos.data(), curVelEnd.data(), BLUE);
    }
#endif
  }

protected:
  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.model_vals_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }
  bool IsFabricsSupported() const override { return true; }
  FabPlannerConfigPtr GetFabricsConfig() const override;

private:
  ResidualFn residual_;
  bool is_static = false;
  DDGBodyPtr mug_ = nullptr;
};
}  // namespace mjpc::manipulation

#endif  // MJPC_MJPC_TASKS_MANIPULATION_MANIPULATION_H_
