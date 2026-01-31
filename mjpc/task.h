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

#ifndef MJPC_TASK_H_
#define MJPC_TASK_H_

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define MJPC_PLANNER_IDTO_ENABLED (0)

#if MJPC_PLANNER_IDTO_ENABLED
// drake
#include <drake/geometry/meshcat.h>
#include <drake/multibody/plant/multibody_plant.h>
#endif

// mjpc
#include "mjpc/norm.h"
#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_config.h"
#include "mjpc/planners/fabrics/include/fab_goal.h"
#if MJPC_PLANNER_IDTO_ENABLED
#include "mjpc/planners/idto/idto_common.h"
#include "mjpc/planners/idto/idto_yaml_config.h"
#endif
#include "mjpc/planners/rmp/include/core/rmp_state.h"
#include "mjpc/planners/rmp/include/util/rmp_util.h"
#include "mjpc/utilities.h"

namespace mjpc
{
  // tolerance for risk-neutral cost
  inline constexpr double kRiskNeutralTolerance = 1.0e-6;

  // maximum cost terms
  inline constexpr int kMaxCostTerms = 128;

  class Task;
  class Planner;

#define MJPC_LOCK_TASK_DATA_ACCESS std::lock_guard<std::mutex> lock0(task_data_mutex_);

  // abstract class for a residual function
  class AbstractResidualFn
  {
  public:
    virtual ~AbstractResidualFn() = default;

    virtual void Residual(const mjModel* model, const mjData* data, double* residual) const = 0;
    virtual void CostTerms(double* terms, const double* residual, bool weighted) const = 0;
    virtual double CostValue(const double* residual) const = 0;

    // copies weights and parameters from the Task instance. This should be
    // called from the Task class.
    virtual void Update() = 0;
  };

  // base implementation for ResidualFn implementations
  class BaseResidualFn : public AbstractResidualFn
  {
  public:
    explicit BaseResidualFn(const Task* task);
    ~BaseResidualFn() override = default;

    void CostTerms(double* terms, const double* residual, bool weighted) const override;
    double CostValue(const double* residual) const override;
    void Update() override;

  protected:
    int num_residual_;
    int num_term_;
    int num_trace_;
    std::vector<int> dim_norm_residual_;
    std::vector<int> num_norm_parameter_;
    std::vector<NormType> norm_;
    std::vector<double> weight_;
    std::vector<double> norm_parameter_;
    double risk_;
    std::vector<double> parameters_;
    const Task* task_;
  };

  // Thread-safe interface for classes that implement MJPC task specifications
  class Task
  {
  public:
    // constructor
    Task() = default;
    virtual ~Task() = default;

    void Initialize(mjModel* model)
    {
      model_ = model;

      // Init planners initial specifics (that must be done main-thread, eg: UI)
      InitFabrics();
      InitCIO();
#if MJPC_PLANNER_IDTO_ENABLED
      InitIdto();
#endif
    }

    // Fabrics
    void InitFabrics()
    {
    }

    virtual bool IsFabricsSupported() const { return false; }
    virtual FabPlannerConfigPtr GetFabricsConfig() const { return std::make_shared<FabPlannerConfig>(); }

    // CIO
    void InitCIO()
    {
    }

    virtual bool IsCIOSupported() const { return true; }
    virtual std::vector<double> GetObservationsData(bool with_noise = true) const { return {}; }

#if MJPC_PLANNER_IDTO_ENABLED
    // Idto
    virtual void CreateDrakePlantModel(drake::multibody::MultibodyPlant<double>* plant) const
    {
    }

    void InitIdto();

    virtual void InitMeshcat()
    {
    }

    virtual void UpdateMeshcatFromIdtoConfigs()
    {
    }

    IdtoPlannerConfigPtr idto_configs_ = nullptr;
    std::string idto_configs_path_;
    static DrakeMeshcatPtr Meshcat() { return meshcat_; }
    static DrakeMeshcatPtr meshcat_;
#endif

    // Bimanual
    virtual bool IsBimanualSupported() const { return false; }

    // Lsqp
    virtual bool IsLSQPSupported() const { return false; }

    // Override model
    UniqueMjModel ComposeOverrideModel()
    {
      mjModel* model = ConstructModel();
      model_programmingly_built_ = static_cast<bool>(model);
      if (model)
      {
        // NOTE: Since as in [Agent::Initialize()], this override custom-constructed model is used for the planner first then copied to all tasks later,
        // -> [Configure(model)] cannot be put inside [::Initialize(model)]
        ConfigureModel(model);
        return {model, mj_deleteModel};
      }
      return {nullptr, nullptr};
    }

    // Programmingly construct model
    virtual mjModel* ConstructModel()
    {
      return nullptr;
    }

    // Customizingly configure model (timestep, gravity, etc.)
    virtual void ConfigureModel(mjModel* model)
    {
    }

    // delegates to ResidualLocked, while holding a lock
    std::unique_ptr<AbstractResidualFn> Residual() const;

    // ----- methods ----- //
    // calls Residual on the pointer returned from InternalResidual(), while
    // holding a lock
    void Residual(const mjModel* model, const mjData* data, double* residual) const;

    // Must be called whenever parameters or weights change outside Transition or
    // Reset, so that calls to Residual use the new parameters.
    // Calls InternalResidual()->Update() with a lock.
    void UpdateResidual();

    // Changes to data will affect the planner at the next set_state.  Changes to
    // model will only affect the physics and render threads, and will not affect
    // the planner. This is useful for studying planning under model discrepancy,
    // calls TransitionLocked and InternalResidual()->Update() while holding a
    // lock
    void Transition(mjModel* model, mjData* data);

    // get information from model
    // calls ResetLocked and InternalResidual()->Update() while holding a lock
    void Reset(const mjModel* model);

    // calls CostTerms on the pointer returned from InternalResidual(), while
    // holding a lock
    void CostTerms(double* terms, const double* residual) const;

    // calls CostTerms on the pointer returned from InternalResidual(), while
    // holding a lock
    void UnweightedCostTerms(double* terms, const double* residual) const;

    // calls CostValue on the pointer returned from InternalResidual(), while
    // holding a lock
    virtual double CostValue(const double* residual) const;

    virtual void ModifyScene(const mjModel* model, const mjData* data, mjvScene* scene) const
    {
    }

    virtual std::string Name() const = 0;
    virtual std::string XmlPath() const = 0;
    virtual std::string RobotModelPath() const { return {}; }
    virtual std::string GetBaseBodyName() const { return {}; }

    virtual std::vector<std::string> GetEndtipNames() const
    {
      /* Ones in URDF, not XML */
      return {};
    }

    virtual std::vector<std::string> GetCollisionLinkNames() const
    {
      /* Ones in URDF, not XML */
      return {};
    }

    virtual FabSelfCollisionNamePairs GetSelfCollisionNamePairs() const
    {
      /* Ones in URDF, not XML */
      return {};
    }

    virtual FabLinkCollisionProps GetCollisionLinkProps() const { return {}; }
    virtual std::vector<FabJointLimit> GetJointLimits() const { return {}; }
    virtual std::vector<FabSubGoalPtr> GetSubGoals() const { return {}; }

    virtual int GetActionDim() const { return 0; }
    virtual int GetTargetObjectId() const { return -1; }
    virtual int GetTargetObjectGeomId() const { return -1; }

    const mjtNum* QueryTargetPos(bool inertia_com = true) const
    {
      return QueryBodyPos(GetTargetObjectId(), inertia_com);
    }

    const mjtNum* QueryTargetQuat(bool inertia_com = true) const
    {
      return QueryBodyQuat(GetTargetObjectId(), inertia_com);
    }

    const mjtNum* QueryTargetVel(bool linear = true) const { return QueryBodyVel(GetTargetObjectId(), linear); }
    const mjtNum* QueryTargetAcc(bool linear = true) const { return QueryBodyAcc(GetTargetObjectId(), linear); }

    virtual bool CheckBlocking(const double start[], const double end[]) { return false; }

    // model
    mjModel* model_ = nullptr;
    mjData* data_ = nullptr;
    Planner* planner_ = nullptr;
    mjvScene* scene_ = nullptr;
    bool model_programmingly_built_ = false;

    bool IsModelProgramminglyBuilt() const { return model_programmingly_built_; }
    virtual const mjtNum* GetRobotPos() const { return QueryTargetPos(); }
    virtual const mjtNum* GetRobotVel() const { return QueryTargetVel(); }
    virtual const mjtNum* GetRobotAcc() const { return QueryTargetAcc(); }
    virtual const mjtNum* GetGoalPos() const { return nullptr; }
    virtual const mjtNum* GetGoalQuat() const { return nullptr; }
    virtual const mjtNum* GetGoalVel() const { return nullptr; }
    virtual const mjtNum* GetGoalAcc() const { return nullptr; }

    // Joint
    int QueryJointId(const char* joint_name) const
    {
      return model_ ? mj_name2id(model_, mjOBJ_JOINT, joint_name) : -1;
    }

    int QueryJointPosAddress(const char* joint_name) const
    {
      int joint_id = QueryJointId(joint_name);
      return (model_ && (joint_id >= 0) && (joint_id < model_->njnt)) ? model_->jnt_qposadr[joint_id] : 0;
    }

    int QueryJointDofAddress(const char* joint_name) const
    {
      int joint_id = QueryJointId(joint_name);
      return (model_ && (joint_id >= 0) && (joint_id < model_->njnt)) ? model_->jnt_dofadr[joint_id] : 0;
    }

    // NOTE: model_->nq,nv are actuated joints/controls configured in MJ model
    // dof: full dof of the robot
    std::vector<double> QueryJointPositions(int dof, const std::string& first_joint_name = {}) const
    {
      if (model_ && data_)
      {
        std::vector<double> qpos(dof, 0);
        const auto& joint_name = first_joint_name.empty() ? first_joint_name_ : first_joint_name;
        mju_copy(qpos.data(), data_->qpos + QueryJointPosAddress(joint_name.c_str()),
                 std::min(int(model_->nq), dof));
        return qpos;
      }
      return {};
    }

    std::vector<double> QueryJointVels(int dof, const std::string& first_joint_name = {}) const
    {
      if (model_ && data_)
      {
        std::vector<double> qvel(dof, 0);
        const auto& joint_name = first_joint_name.empty() ? first_joint_name_ : first_joint_name;
        mju_copy(qvel.data(), data_->qvel + QueryJointDofAddress(joint_name.c_str()),
                 std::min(int(model_->nv), dof));
        return qvel;
      }
      return {};
    }

    virtual void SetPlanner(Planner* planner) { planner_ = planner; }
    Planner* GetPlanner() const { return planner_; }

    // Body
    int QueryBodyId(const char* body_name) const
    {
      return mjpc::QueryBodyId(model_, body_name);
    }

    mjtNum* QueryBodyQuat(int body_id, bool inertia_com = true) const
    {
      return mjpc::QueryBodyQuat(data_, body_id, inertia_com);
    }

    mjtNum* QueryBodyQuat(const char* body_name, bool inertia_com = true) const
    {
      return QueryBodyQuat(QueryBodyId(body_name), inertia_com);
    }

    mjtNum* QueryBodyRotMat(int body_id, bool inertia_com = true) const
    {
      return mjpc::QueryBodyRotMat(data_, body_id, inertia_com);
    }

    mjtNum* QueryBodyRotMat(const char* body_name, bool inertia_com = true) const
    {
      return QueryBodyRotMat(QueryBodyId(body_name), inertia_com);
    }

    mjtNum* QueryBodyPos(int body_id, bool inertia_com = true) const
    {
      return mjpc::QueryBodyPos(data_, body_id, inertia_com);
    }

    mjtNum* QueryBodyPos(const char* body_name, bool inertia_com = true) const
    {
      return QueryBodyPos(QueryBodyId(body_name), inertia_com);
    }

    mjtNum* QueryBodyVel(int body_id, bool linear = true) const
    {
      return mjpc::QueryBodyVel(data_, body_id, linear);
    }

    mjtNum* QueryBodyVel(const char* body_name, bool inertia_com = true) const
    {
      return QueryBodyVel(QueryBodyId(body_name), inertia_com);
    }

    mjtNum* QueryBodyAcc(int body_id, bool linear = true) const
    {
      return mjpc::QueryBodyAcc(data_, body_id, linear);
    }

    mjtNum* QueryBodyAcc(const char* body_name, bool inertia_com = true) const
    {
      return QueryBodyAcc(QueryBodyId(body_name), inertia_com);
    }

    // Body mocap
    int QueryBodyMocapId(const char* body_name) const
    {
      return mjpc::QueryBodyMocapId(model_, body_name);
    }

    void SetBodyMocapPos(const char* body_name, const double* pos) const
    {
      mjpc::SetBodyMocapPos(model_, data_, body_name, pos);
    }

    void MoveBodyMocapToSite(const char* body_name, const char* site_name) const
    {
      mjpc::MoveBodyMocapToSite(model_, data_, body_name, site_name);
    }

    void MoveBodyMocapToSite(const char* body_name, int site_id) const
    {
      mjpc::MoveBodyMocapToSite(model_, data_, body_name, site_id);
    }

    mjtNum* QueryBodyMocapPos(const char* body_name) const
    {
      return mjpc::QueryBodyMocapPos(model_, data_, body_name);
    }

    void SetBodyMocapQuat(const char* body_name, const double* quat) const
    {
      mjpc::SetBodyMocapQuat(model_, data_, body_name, quat);
    }

    mjtNum* QueryBodyMocapQuat(const char* body_name) const
    {
      return mjpc::QueryBodyMocapQuat(model_, data_, body_name);
    }

    mjtNum QueryBodyMass(int body_id) const
    {
      return mjpc::QueryBodyMass(model_, body_id);
    }

    mjtNum QueryBodyMass(const char* body_name) const
    {
      return mjpc::QueryBodyMass(model_, body_name);
    }

    // Geom
    int QueryGeomId(const char* geom_name) const
    {
      return mjpc::QueryGeomId(model_, geom_name);
    }

    mjtNum* QueryGeomPos(const char* geom_name) const
    {
      return mjpc::QueryGeomPos(model_, data_, geom_name);
    }

    mjtNum* QueryGeomQuat(const char* geom_name) const
    {
      return mjpc::QueryGeomQuat(model_, data_, geom_name);
    }

    std::vector<double> QueryGeomSize(int geom_id) const
    {
      return mjpc::QueryGeomSize(model_, geom_id);
    }

    std::vector<double> QueryGeomSize(const char* site_name) const
    {
      return QueryGeomSize(QueryGeomId(site_name));
    }

    double QueryGeomSizeMax(const char* geom_name) const
    {
      return mjpc::QueryGeomSizeMax(model_, geom_name);
    }

    double QueryGeomSizeMax(const std::vector<const char*>& geom_name_list) const
    {
      return mjpc::QueryGeomSizeMax(model_, geom_name_list);
    }

    void SetGeomColor(uint geom_id, const float* rgba) const
    {
      if (scene_ && (geom_id < model_->ngeom))
      {
        memcpy(scene_->geoms[geom_id].rgba, rgba, sizeof(float) * 4);
      }
    }

    // Site
    int QuerySiteId(const char* site_name) const
    {
      return mjpc::QuerySiteId(model_, site_name);
    }

    mjtNum* QuerySitePos(int site_id) const
    {
      return mjpc::QuerySitePos(data_, site_id);
    }

    mjtNum* QuerySitePos(const char* site_name) const
    {
      return mjpc::QuerySitePos(model_, data_, site_name);
    }

    mjtNum* QuerySiteQuat(int site_id) const
    {
      return mjpc::QuerySiteQuat(data_, site_id);
    }

    mjtNum* QuerySiteQuat(const char* site_name) const
    {
      return mjpc::QuerySiteQuat(model_, data_, site_name);
    }

    std::vector<double> QuerySiteSize(int site_id) const
    {
      return mjpc::QuerySiteSize(model_, site_id);
    }

    std::vector<double> QuerySiteSize(const char* site_name) const
    {
      return QuerySiteSize(QuerySiteId(site_name));
    }

    double QuerySiteSizeMax(const char* site_name) const
    {
      return mjpc::QuerySiteSizeMax(model_, site_name);
    }

    // mode
    int mode;

    // GUI toggles
    int reset = 0;
    int visualize = 0;

    // cost parameters
    int num_residual;
    int num_term;
    int num_trace;
    std::vector<int> dim_norm_residual;
    std::vector<int> num_norm_parameter;
    std::vector<NormType> norm;
    std::vector<double> weight;
    std::vector<double> norm_parameter;
    double risk;

    // residual parameters
    std::vector<double> parameters;

    // universal raytraces
    std::vector<mjtNum> ray_starts;
    std::vector<mjtNum> ray_ends;
    bool last_goal_reached_ = false;

    // RMP/Fabrics --
    using StateX = rmp::State<3>;
    std::vector<StateX> obstacle_statesX_;

    // mutex which should be held on changes to data queried from mjdata
    mutable std::mutex task_data_mutex_;

    // - Actuator/Joint
    float actuator_kv = 1.f;
    std::string first_joint_name_;
    FabControlMode fabrics_control_mode_ = FabControlMode::VEL;
    FabControlMode GetFabricsControlMode() const { return fabrics_control_mode_; }

    // - Goal
    FabDynamicsState goal_state_ =
      FabDynamicsState{.default_lin = std::vector(3, 0.), .default_ang = std::vector(3, 0.)};
    virtual bool IsGoalFixed() const { return true; }

    virtual bool QueryGoalReached()
    {
      auto* goal_pos = GetGoalPos();
      return goal_pos &&
        (rmp::vectorFromScalarArray<3>(GetRobotPos()) - rmp::vectorFromScalarArray<3>(goal_pos)).norm() <
        0.005;
    }

    Eigen::Vector3d rotMatrixToEulerAngles(Eigen::Matrix3d& R) const
    {
      return R.eulerAngles(0, 1, 2); // XYZ or RPY
    }

    virtual void QueryGoalState()
    {
      MJPC_LOCK_TASK_DATA_ACCESS;
      auto* goal_pos = GetGoalPos();
      if (goal_pos)
      {
        goal_state_.reset();
        mju_copy3(goal_state_.pose.pos.data(), goal_pos);
#if 0
        mjtNum mat[9];
        mju_quat2Mat(mat, GetGoalQuat());
        Eigen::Matrix3d emat;
        mju_copy(emat.data(), mat, 9 * sizeof(double));
        const auto erot = rotMatrixToEulerAngles(emat);
        mjtNum rot[3];
        mju_copy3(rot, erot.data());
        mju_copy3(goal_state_.pose.rot.data(), rot);
#endif
        mju_copy3(goal_state_.linear_vel.data(), GetGoalVel());
        mju_copy3(goal_state_.linear_acc.data(), GetGoalAcc());
      }
    }

    FabDynamicsState GetGoalState() const
    {
      FabDynamicsState goal_state;
      {
        MJPC_LOCK_TASK_DATA_ACCESS;
        goal_state = goal_state_;
      }
      return goal_state;
    }

    // - Obstacles
    virtual bool AreObstaclesFixed() const { return (GetDynamicObstaclesNum() == 0); }
    int GetObstaclesDim() const { return AreObstaclesFixed() ? 3 : GetDynamicObstaclesDimension(); }
    virtual int GetDynamicObstaclesDimension() const { return 3; }

    virtual void QueryObstacleStatesX()
    {
    }

    std::vector<StateX> GetObstacleStatesX() const
    {
      std::vector<StateX> obstacle_statesX;
      {
        MJPC_LOCK_TASK_DATA_ACCESS;
        obstacle_statesX = obstacle_statesX_;
      }
      return obstacle_statesX;
    }

    int static_obstacles_num = 0;
    int dynamic_obstacles_num = 0;
    virtual int GetStaticObstaclesNum() const { return static_obstacles_num; }
    virtual int GetDynamicObstaclesNum() const { return dynamic_obstacles_num; }
    int GetTotalObstaclesNum() const { return GetStaticObstaclesNum() + GetDynamicObstaclesNum(); }

    // - Constraints
    virtual int GetPlaneConstraintsNum() const { return 0; }

  protected:
    // returns a pointer to the ResidualFn instance that's used for physics
    // stepping and plotting, and is internal to the class
    virtual BaseResidualFn* InternalResidual() = 0;
    const BaseResidualFn* InternalResidual() const { return const_cast<Task*>(this)->InternalResidual(); }
    // returns an object which can compute the residual function. the function
    // can assume that a lock on mutex_ is held when it's called
    virtual std::unique_ptr<AbstractResidualFn> ResidualLocked() const = 0;
    // implementation of Task::Transition() which can assume a lock is held.
    // in some cases the transition logic requires calling mj_forward (e.g., for
    // measuring contact forces), which will call the sensor callback, which calls
    // ResidualLocked. In order to avoid such resource contention, mutex_ might be
    // temporarily unlocked, but it must be locked again before returning.
    virtual void TransitionLocked(mjModel* model, mjData* data);

    // implementation of Task::Reset() which can assume a lock is held
    virtual void ResetLocked(const mjModel* model)
    {
      model_programmingly_built_ = false;
    }

    // mutex which should be held on changes to InternalResidual.
    mutable std::mutex mutex_;

  private:
    // initial residual parameters from model
    void SetFeatureParameters(const mjModel* model);
  };
} // namespace mjpc

#endif  // MJPC_TASK_H_
