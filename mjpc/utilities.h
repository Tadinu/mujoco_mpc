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

#ifndef MJPC_UTILITIES_H_
#define MJPC_UTILITIES_H_

#include <absl/container/flat_hash_map.h>
#include <omp.h>

#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

// Abseil
#include "absl/container/btree_set.h"
#include "absl/types/span.h"

// mujoco
#include <mujoco/mujoco.h>
#include "mjpc/utils/mjpc_core_util.h"
#include "mjpc/utils/mjpc_math_util.h"

#define MJPC_OPENMP_ENABLED (1)
#define MJPC_OPENMP_THREADS_NUM (1000)

namespace mjpc
{
  // maximum number of traces that are visualized
  inline constexpr int kMaxTraces = 99;

  // make model differentiable by setting solimp[0] to zero
  void MakeDifferentiable(mjModel* model);

  inline bool IsSelectedControl(const std::vector<int>& indices, int idx)
  {
    return indices.empty() || std::find(indices.begin(), indices.end(), idx) != indices.end();
  }

  // Spec
  inline mjsBody* FindWorldBodySpec(mjSpec* model_spec)
  {
    return mjs_findBody(model_spec, "world");
  }

  inline mjsKey* FindKeySpec(mjSpec* model_spec, const char* key_name)
  {
    return mjs_asKey(mjs_findElement(model_spec, mjOBJ_KEY, key_name));
  }

  inline mjsSite* FindSiteSpec(mjSpec* model_spec, const char* site_name)
  {
    return mjs_asSite(mjs_findElement(model_spec, mjOBJ_SITE, site_name));
  }

  // find all child specs of either a spec or a body
  // Ref: https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/specs.cc - FindAllImpl
  template <typename TSpec, typename TChildSpec,
            typename = std::enable_if<std::is_same_v<TSpec, mjSpec> | std::is_same_v<TSpec, mjsBody>>>
  inline std::vector<TChildSpec*> FindAllChildSpecs(TSpec* base_spec, mjtObj type, int recurse)
  {
    if (type == mjOBJ_UNKNOWN)
    {
      // this should never happen
      throw MjpcError(
        "[FindAllChildSpecs] supports the types: body, frame, geom, site, "
        "joint, light, camera.");
    }

    std::vector<TChildSpec*> list;
    mjsElement* el = nullptr;
    if constexpr (std::is_same_v<TSpec, mjSpec>)
    {
      el = mjs_firstElement(base_spec, type);
    }
    else
    {
      el = mjs_firstChild(base_spec, type, recurse);
    }

    const std::string error = mjs_getError(mjs_getSpec(base_spec->element));
    if (!el && !error.empty())
    {
      throw MjpcError(error);
    }
    while (el)
    {
      if constexpr (std::is_same_v<TChildSpec, mjsElement>)
      {
        list.push_back(el);
      }
      else
      {
        TChildSpec* child_spec = nullptr;
        if constexpr (std::is_same_v<TChildSpec, mjsBody>)
        {
          child_spec = mjs_asBody(el);
        }
        else if constexpr (std::is_same_v<TChildSpec, mjsCamera>)
        {
          child_spec = mjs_asCamera(el);
        }
        else if constexpr (std::is_same_v<TChildSpec, mjsFrame>)
        {
          child_spec = mjs_asFrame(el);
        }
        else if constexpr (std::is_same_v<TChildSpec, mjsGeom>)
        {
          child_spec = mjs_asGeom(el);
        }
        else if constexpr (std::is_same_v<TChildSpec, mjsJoint>)
        {
          child_spec = mjs_asJoint(el);
        }
        else if constexpr (std::is_same_v<TChildSpec, mjsLight>)
        {
          child_spec = mjs_asLight(el);
        }
        else if constexpr (std::is_same_v<TChildSpec, mjsSite>)
        {
          child_spec = mjs_asSite(el);
        }

        if (child_spec)
        {
          list.push_back(child_spec);
        }
      }

      if constexpr (std::is_same_v<TSpec, mjSpec>)
      {
        el = mjs_nextElement(base_spec, el);
      }
      else
      {
        el = mjs_nextChild(base_spec, el, recurse);
      }
    }
    return list;
  }

  // Joint
  inline int QueryJointIdFromDof(const mjModel* model, const int dof_id)
  {
    return model ? model->dof_jntid[dof_id] : -1;
  }

  inline int QueryJointId(const mjModel* model, const char* joint_name)
  {
    return model ? mj_name2id(model, mjOBJ_JOINT, joint_name) : -1;
  }

  inline int QueryJointPosAddress(const mjModel* model, int joint_id)
  {
    return (model && (joint_id > -1) && (joint_id < model->njnt)) ? model->jnt_qposadr[joint_id] : -1;
  }

  inline int QueryJointPosAddress(const mjModel* model, const char* joint_name)
  {
    return QueryJointPosAddress(model, QueryJointId(model, joint_name));
  }

  inline int QueryJointDofAddress(const mjModel* model, int joint_id)
  {
    return (model && (joint_id > -1) && (joint_id < model->njnt)) ? model->jnt_dofadr[joint_id] : -1;
  }

  inline int QueryJointDofAddress(const mjModel* model, const char* joint_name)
  {
    return QueryJointDofAddress(model, QueryJointId(model, joint_name));
  }

  // NOTE: model_->nq,nv are actuated joints/controls configured in MJ model
  // This applies only for continuous-ids joints
  // dof_num: full dof of the robot
  inline std::vector<double> QueryJointPositions(const mjModel* model, const mjData* data, int dof_num = -1,
                                                 const std::string& first_joint_name = {})
  {
    if (model && data)
    {
      if (dof_num < 0)
      {
        dof_num = model->nq;
      }
      std::vector<double> qpos(dof_num, 0);
      mju_copy(qpos.data(),
               data->qpos + (first_joint_name.empty()
                               ? 0
                               : QueryJointPosAddress(model, first_joint_name.c_str())),
               std::min(int(model->nq), dof_num));
      return qpos;
    }
    return {};
  }

  inline Eigen::VectorXd QueryJointPositionsEigen(const mjModel* model, const mjData* data, int dof_num = -1,
                                                  const std::string& first_joint_name = {})
  {
    std::vector<double> qpos_vec = mjpc::QueryJointPositions(model, data);
    return Eigen::Map<Eigen::VectorXd>(qpos_vec.data(), qpos_vec.size());
  }

  inline double QuerySingleJointPos(const mjModel* model, const mjData* data, int jnt_id)
  {
    if (model && data && (jnt_id > -1))
    {
      assert(jnt_id < model->nq);
      return data->qpos[jnt_id];
    }
    return 0.;
  }

  // NOTE: This applies only for continous-ids joints
  inline std::vector<double> QueryJointVels(const mjModel* model, const mjData* data, int dof_num,
                                            const std::string& first_joint_name = {})
  {
    if (model && data)
    {
      std::vector<double> qvel(dof_num, 0);
      mju_copy(qvel.data(), data->qvel + (first_joint_name.empty()
                                            ? 0
                                            : QueryJointDofAddress(model, first_joint_name.c_str())),
               std::min(int(model->nv), dof_num));
      return qvel;
    }
    return {};
  }

  inline double QuerySingleJointVel(const mjModel* model, const mjData* data, int dof_id)
  {
    if (model && data && (dof_id > -1))
    {
      assert(dof_id < model->nv);
      return data->qvel[dof_id];
    }
    return 0.;
  }

  inline void PrintJoints(const mjModel* model, const mjData* data)
  {
    for (int i = 0; i < model->njnt; ++i)
    {
      int name_jntadr = model->name_jntadr[i];
      mjpc::print(std::string(model->names + name_jntadr), mjpc::QuerySingleJointPos(model, data, i));
    }
  }

  inline void PrintDofs(const mjModel* model, const mjData* data)
  {
    for (int i = 0; i < model->nv; ++i)
    {
      int body_id = model->dof_bodyid[i];
      int name_bodyadr = model->name_bodyadr[body_id];
      int jnt_id = model->dof_jntid[i];
      int name_jntadr = model->name_jntadr[jnt_id];
      mjpc::print(std::string(model->names + name_bodyadr), std::string(model->names + name_jntadr),
                  mjpc::QuerySingleJointPos(model, data, jnt_id));
    }
  }

  // Dof
  // Return joint's first dof_id
  inline int QueryDofId(const mjModel* model, const char* jnt_name)
  {
    // NOTE: There is no such [name_dofadr], so mj_name2id(model, mjOBJ_DOF, dof_name) does not work!
    if (model)
    {
#if 1
      const int jnt_id = mj_name2id(model, mjOBJ_JOINT, jnt_name);
      return (jnt_id >= 0) ? model->jnt_dofadr[jnt_id] : -1;
#else
      for (int i = 0; i < model->nv; ++i)
      {
        if (std::string(mj_id2name(model, mjOBJ_JOINT, model->dof_jntid[i])) == jnt_name)
        {
          return i;
        }
      }
#endif
    }
    return -1;
  }

  inline int QueryDofId(const mjModel* model, int jnt_id)
  {
    return QueryDofId(model, mj_id2name(model, mjOBJ_JOINT, jnt_id));
  }

  // Return body's first dof_id
  inline int QueryDofIdFromBody(const mjModel* model, int body_id)
  {
    return (model && (body_id >= 0)) ? model->body_dofadr[body_id] : -1;
  }

  inline int QueryDofIdFromBody(const mjModel* model, const char* body_name);

  // Actuator
  inline int QueryActuatorId(const mjModel* model, const char* actuator_name)
  {
    return model ? mj_name2id(model, mjOBJ_ACTUATOR, actuator_name) : -1;
  }

  // Key
  inline int QueryKeyId(const mjModel* model, const std::string& key_name)
  {
    return model ? mj_name2id(model, mjOBJ_KEY, key_name.c_str()) : -1;
  }

  inline std::vector<double> QueryKeyJointPositions(const mjModel* model, const std::string& key_name)
  {
    if (model)
    {
      const int key_id = QueryKeyId(model, key_name);
      auto* key_pos = &model->key_qpos[key_id * model->nq];
      return (key_id > -1) ? std::vector(key_pos, key_pos + model->nq) : std::vector<double>{};
    }
    return {};
  }

  // Body
  inline int QueryBodyId(const mjModel* model, const char* body_name)
  {
    return model ? mj_name2id(model, mjOBJ_BODY, body_name) : -1;
  }

  inline mjtNum* QueryBodyQuat(const mjData* data, int body_id, bool inertia_com = true)
  {
    if (data && (body_id > -1))
    {
      if (inertia_com)
      {
        static mjtNum quat[4];
        mju_mat2Quat(quat, &data->ximat[9 * body_id]);
        return &quat[0];
      }
      else
      {
        return &data->xquat[4 * body_id];
      }
    }
    return nullptr;
  }

  inline mjtNum* QueryBodyQuat(const mjModel* model, const mjData* data, const char* body_name,
                               bool inertia_com = true)
  {
    return QueryBodyQuat(data, QueryBodyId(model, body_name), inertia_com);
  }

  inline mjtNum* QueryBodyRotMat(const mjData* data, int body_id, bool inertia_com = true)
  {
    if (data && (body_id > -1))
    {
      if (inertia_com)
      {
        return &data->ximat[9 * body_id];
      }
      else
      {
        static mjtNum mat[9];
        mju_quat2Mat(mat, &data->xquat[4 * body_id]);
        return &mat[0];
      }
    }
    return nullptr;
  }

  inline mjtNum* QueryBodyRotMat(const mjModel* model, const mjData* data, const char* body_name,
                                 bool inertia_com = true)
  {
    return QueryBodyRotMat(data, QueryBodyId(model, body_name), inertia_com);
  }

  inline mjtNum* QueryBodyPos(const mjData* data, int body_id, bool inertia_com = true)
  {
    if (data && (body_id > -1))
    {
      return inertia_com ? &data->xipos[3 * body_id] : &data->xpos[3 * body_id];
    }
    return nullptr;
  }

  inline mjtNum* QueryBodyPos(const mjModel* model, const mjData* data, const char* body_name,
                              bool inertia_com = true)
  {
    return QueryBodyPos(data, QueryBodyId(model, body_name), inertia_com);
  }

  inline Eigen::Vector3d QueryBodyPosEigen(const mjData* data, int body_id, bool inertia_com = true)
  {
    auto* pos = QueryBodyPos(data, body_id, inertia_com);
    return pos ? Eigen::Vector3d(Eigen::Map<Eigen::Vector3d>(pos, 3)) : Eigen::Vector3d::Zero();
  }

  inline std::pair<Eigen::Vector3d, Eigen::Quaterniond> QueryBodyPoseEigen(
    const mjModel* model, const mjData* data,
    const std::string& child_body_name,
    const std::string& parent_body_name = {})
  {
    std::pair<Eigen::Vector3d, Eigen::Quaterniond> res;
    mjtNum parent_pose[7];
    mju_copy(parent_pose, POSE_IDENTITY, 7);
    if (!parent_body_name.empty())
    {
      auto parent_body_id = QueryBodyId(model, parent_body_name.c_str());
      mju_copy3(&parent_pose[0], QueryBodyPos(data, parent_body_id));
      mju_copy3(&parent_pose[3], QueryBodyQuat(data, parent_body_id));
    }

    mjtNum rel_pose[7];
    mju_copy(rel_pose, POSE_IDENTITY, 7);
    auto child_body_id = QueryBodyId(model, child_body_name.c_str());
    // TODO: Check mj_local2Global() can be used here or not
#if 0
    mj_local2Global(data, QueryBodyPos(data, child_body_id), QueryBodyRotMat(data, child_body_id, true),
                    QueryBodyPos(data, child_body_id, true), QueryBodyQuat(data, child_body_id, true),
                    child_body_id, model->body_sameframe[child_body_id]);
#endif

    MjuLocalPos(&rel_pose[0], QueryBodyPos(data, child_body_id), &parent_pose[0],
                &parent_pose[3]);
    MjuLocalQuat(&rel_pose[3], QueryBodyQuat(data, child_body_id), &parent_pose[3]);

    mju_copy3(res.first.data(), &rel_pose[0]);
    res.second = mjpc::QuatToEigen(&rel_pose[3]);
    return res;
  }

  inline mjtNum* QueryBodyVel(const mjData* data, int body_id, bool linear = true)
  {
    if (data && (body_id > -1))
    {
      static double lvel[3] = {0};
#if 1
      mju_copy3(lvel, linear ? &data->cvel[6 * body_id + 3] : &data->cvel[6 * body_id]);
#else
      mjtNum vel[6];
      mj_objectVelocity(model_, data_, mjOBJ_BODY, body_id, vel, 0);
      mju_copy3(lvel, linear ? &vel[3] : &vel[0]);
#endif
      return &lvel[0];
    }
    return nullptr;
  }

  inline mjtNum* QueryBodyAcc(const mjData* data, int body_id, bool linear = true)
  {
    if (data && (body_id > -1))
    {
      static double lacc[3] = {0};
#if 1
      mju_copy3(lacc, linear ? &data->cacc[6 * body_id + 3] : &data->cacc[6 * body_id]);
#else
      mjtNum acc[6];
      mj_objectAcceleration(model_, data_, mjOBJ_BODY, body_id, acc, 0);
      mju_copy3(lacc, linear ? &acc[3] : &acc[0]);
#endif
      return &lacc[0];
    }
    return nullptr;
  }

  // Recursive function to set collision properties for a body and its descendants
  inline void SetBodyTreeCollisionEnabled(mjsBody* base_body_spec, bool enabled)
  {
    for (const auto& geom_spec : FindAllChildSpecs<mjsBody, mjsGeom>(base_body_spec, mjOBJ_GEOM, true))
    {
      geom_spec->contype = enabled;
      geom_spec->conaffinity = enabled;
    }
    for (const auto& child_body_spec : FindAllChildSpecs<
           mjsBody, mjsBody>(base_body_spec, mjOBJ_BODY, true))
    {
      SetBodyTreeCollisionEnabled(child_body_spec, enabled);
    }
  }

  // Recursive function to set gravity compensation properties for a body and its descendants
  inline void SetBodyTreeGravityCompensationEnabled(mjsBody* base_body_spec, bool enabled)
  {
    base_body_spec->gravcomp = enabled;
    for (const auto& child_body_spec : FindAllChildSpecs<
           mjsBody, mjsBody>(base_body_spec, mjOBJ_BODY, true))
    {
      SetBodyTreeGravityCompensationEnabled(child_body_spec, enabled);
    }
  }

  // Body mocap
  inline int QueryBodyMocapId(const mjModel* model, const char* body_name)
  {
    if (model)
    {
      int body_id = QueryBodyId(model, body_name);
      return (body_id > -1) ? model->body_mocapid[body_id] : -1;
    }
    return -1;
  }

  inline void SetBodyMocapPos(const mjModel* model, const mjData* data, const char* body_name,
                              const double* pos)
  {
    if (data)
    {
      int bodyMocapId = QueryBodyMocapId(model, body_name);
      if (bodyMocapId > -1)
      {
        mju_copy3(&data->mocap_pos[3 * bodyMocapId], pos);
      }
    }
  }

  inline mjtNum* QueryBodyMocapPos(const mjModel* model, const mjData* data, const char* body_name)
  {
    if (data)
    {
      int bodyMocapId = QueryBodyMocapId(model, body_name);
      return (bodyMocapId > -1) ? &data->mocap_pos[3 * bodyMocapId] : nullptr;
    }
    return nullptr;
  }

  inline void SetBodyMocapQuat(const mjModel* model, const mjData* data, const char* body_name,
                               const double* quat)
  {
    if (data)
    {
      int bodyMocapId = QueryBodyMocapId(model, body_name);
      if (bodyMocapId > -1)
      {
        mju_copy4(&data->mocap_quat[4 * bodyMocapId], quat);
      }
    }
  }

  inline mjtNum* QueryBodyMocapQuat(const mjModel* model, const mjData* data, const char* body_name)
  {
    if (data)
    {
      int bodyMocapId = QueryBodyMocapId(model, body_name);
      return (bodyMocapId > -1) ? &data->mocap_quat[4 * bodyMocapId] : nullptr;
    }
    return nullptr;
  }

  inline mjtNum QueryBodyMass(const mjModel* model, int body_id)
  {
    if (model)
    {
      return (body_id > -1) ? model->body_mass[body_id] : 0;
    }
    return 0;
  }

  inline mjtNum QueryBodyMass(const mjModel* model, const char* body_name)
  {
    if (model)
    {
      int bodyId = QueryBodyId(model, body_name);
      return (bodyId > -1) ? model->body_mass[bodyId] : 0;
    }
    return 0;
  }

  inline void PrintBodyMocaps(const mjModel* model, const mjData* data)
  {
    for (auto i = 0; i < model->nbody; ++i)
    {
      auto mocap_id = model->body_mocapid[i];
      if (mocap_id >= 0)
      {
        const auto mocap_name = std::string(mj_id2name(model, mjOBJ_BODY, i));
        auto* mocap_pos = QueryBodyMocapPos(model, data, mocap_name.c_str());
        auto* mocap_quat = QueryBodyMocapQuat(model, data, mocap_name.c_str());
        Eigen::Vector3d pos = Eigen::Map<Eigen::Vector3d>(mocap_pos, 3);
        Eigen::Vector4d quat = Eigen::Map<Eigen::Vector4d>(mocap_quat, 4);
        print("mocap body:", i, mocap_name,
              "pos:", pos.transpose(), "quat:", quat.transpose());
      }
    }
  }

  // Geom
  inline int QueryGeomId(const mjModel* model, const char* geom_name)
  {
    return model ? mj_name2id(model, mjOBJ_GEOM, geom_name) : -1;
  }

  inline mjtNum* QueryGeomPos(const mjModel* model, const mjData* data, const char* geom_name)
  {
    if (data)
    {
      const int geom_id = QueryGeomId(model, geom_name);
      return (geom_id > -1) ? &data->geom_xpos[3 * geom_id] : nullptr;
    }
    return nullptr;
  }

  inline mjtNum* QueryGeomQuat(const mjModel* model, const mjData* data, const char* geom_name)
  {
    if (data)
    {
      const int geom_id = QueryGeomId(model, geom_name);
      if (geom_id > -1)
      {
        static mjtNum quat[4];
        mju_mat2Quat(quat, &data->geom_xmat[9 * geom_id]);
        return &quat[0];
      }
    }
    return nullptr;
  }

  inline std::vector<double> QueryGeomSize(const mjModel* model, int geom_id)
  {
    std::vector<double> size(3, 0.0);
    if (geom_id > -1)
    {
      mju_copy3(size.data(), &model->geom_size[3 * geom_id]);
    }
    return size;
  }

  inline std::vector<double> QueryGeomSize(const mjModel* model, const char* geom_name)
  {
    return QueryGeomSize(model, QueryGeomId(model, geom_name));
  }

  inline double QueryGeomSizeMax(const mjModel* model, const char* geom_name)
  {
    const auto size = QueryGeomSize(model, geom_name);
    return std::max({size[0], size[1], size[2]});
  }

  inline double QueryGeomSizeMax(const mjModel* model, const std::vector<const char*>& geom_name_list)
  {
    double max = 0.;
    for (const auto& geom_name : geom_name_list)
    {
      const auto size = QueryGeomSize(model, geom_name);
      max = std::max(max, std::max({size[0], size[1], size[2]}));
    }
    return max;
  }

  inline void SetGeomColor(const mjvScene* scene, const mjModel* model, uint geom_id, const float* rgba)
  {
    if (scene && (geom_id > -1) && (geom_id < model->ngeom))
    {
      memcpy(scene->geoms[geom_id].rgba, rgba, sizeof(float) * 4);
    }
  }

  // Site
  inline int QuerySiteId(const mjModel* model, const char* site_name)
  {
    return model ? mj_name2id(model, mjOBJ_SITE, site_name) : -1;
  }

  inline int QueryBodyIdFromSite(const mjModel* model, const char* site_name)
  {
    const auto site_id = QuerySiteId(model, site_name);
    if (site_id > -1)
    {
      return model->site_bodyid[site_id];
    }
    return -1;
  }

  inline int QueryBodyIdFromSite(const mjModel* model, int site_id)
  {
    if ((site_id > -1) && (site_id < model->nsite))
    {
      return model->site_bodyid[site_id];
    }
    return -1;
  }

  inline mjtNum* QuerySitePos(const mjData* data, int site_id)
  {
    if (data && (site_id > -1))
    {
      return &data->site_xpos[3 * site_id];
    }
    return nullptr;
  }

  inline mjtNum* QuerySitePos(const mjModel* model, const mjData* data, const char* site_name)
  {
    return QuerySitePos(data, QuerySiteId(model, site_name));
  }

  inline Eigen::Vector3d QuerySitePosEigen(const mjData* data, int site_id)
  {
    auto* pos = QuerySitePos(data, site_id);
    return pos ? Eigen::Vector3d(Eigen::Map<Eigen::Vector3d>(pos, 3)) : Eigen::Vector3d::Zero();
  }

  inline mjtNum* QuerySiteQuat(const mjData* data, int site_id)
  {
    if (data)
    {
      static mjtNum quat[4];
      mju_mat2Quat(quat, &data->site_xmat[9 * site_id]);
      return &quat[0];
    }
    return nullptr;
  }

  inline mjtNum* QuerySiteQuat(const mjModel* model, const mjData* data, const char* site_name)
  {
    return QuerySiteQuat(data, QuerySiteId(model, site_name));
  }

  inline std::vector<double> QuerySiteSize(const mjModel* model, int site_id)
  {
    std::vector<double> size(3, 0.0);
    if (model && (site_id > -1))
    {
      mju_copy3(size.data(), &model->site_size[3 * site_id]);
    }
    return size;
  }

  inline std::vector<double> QuerySiteSize(const mjModel* model, const char* site_name)
  {
    return QuerySiteSize(model, QuerySiteId(model, site_name));
  }

  inline double QuerySiteSizeMax(const mjModel* model, const char* site_name)
  {
    const auto size = QuerySiteSize(model, site_name);
    return std::max({size[0], size[1], size[2]});
  }

  inline void MoveBodyMocapToSite(const mjModel* model, const mjData* data,
                                  const char* body_name, const char* site_name)
  {
    SetBodyMocapPos(model, data, body_name, QuerySitePos(model, data, site_name));
    SetBodyMocapQuat(model, data, body_name, QuerySiteQuat(model, data, site_name));
  }

  inline void MoveBodyMocapToSite(const mjModel* model, const mjData* data,
                                  const char* body_name, int site_id)
  {
    SetBodyMocapPos(model, data, body_name, QuerySitePos(data, site_id));
    SetBodyMocapQuat(model, data, body_name, QuerySiteQuat(data, site_id));
  }

  inline void PrintSites(const mjModel* model, const mjData* data)
  {
    for (auto i = 0; i < model->nsite; ++i)
    {
      const auto site_name = std::string(mj_id2name(model, mjOBJ_SITE, i));
      auto* site_pos = QuerySitePos(data, i);
      auto* site_quat = QuerySiteQuat(data, i);
      Eigen::Vector3d pos = Eigen::Map<Eigen::Vector3d>(site_pos, 3);
      Eigen::Vector4d quat = Eigen::Map<Eigen::Vector4d>(site_quat, 4);
      print("site:", i, site_name,
            "pos:", pos.transpose(), "quat:", quat.transpose());
    }
  }

  // set mjData state
  void SetState(const mjModel* model, mjData* data, const double* state);

  // get mjData state
  void GetState(const mjModel* model, const mjData* data, double* state);

  // get numerical data from a custom element in mjModel with the given name
  double* GetCustomNumericData(const mjModel* m, std::string_view name);

  // get text data from a custom element in mjModel with the given name
  char* GetCustomTextData(const mjModel* m, std::string_view name);

  // get a scalar value from a custom element in mjModel with the given name
  template <typename T>
  std::optional<T> GetNumber(const mjModel* m, std::string_view name)
  {
    double* data = GetCustomNumericData(m, name);
    if (data)
    {
      return static_cast<T>(data[0]);
    }
    else
    {
      return std::nullopt;
    }
  }

  // get a single numerical value from a custom element in mjModel, or return the
  // default value if a custom element with the specified name does not exist
  template <typename T>
  T GetNumberOrDefault(T default_value, const mjModel* m, std::string_view name)
  {
    return GetNumber<T>(m, name).value_or(default_value);
  }

  // reinterpret double as int
  int ReinterpretAsInt(double value);

  // reinterpret int64_t as double
  double ReinterpretAsDouble(int64_t value);

  // returns a map from custom field name to the list of valid values for that
  // field
  absl::flat_hash_map<std::string, std::vector<std::string>> ResidualSelectionLists(const mjModel* m);

  // get the string selected in a drop down with the given name, given the value
  // in the residual parameters vector
  std::string ResidualSelection(const mjModel* m, std::string_view name, double residual_parameter);
  // returns a value for residual parameters that fits the given text value
  // in the given list
  double ResidualParameterFromSelection(const mjModel* m, std::string_view name, std::string_view value);

  // returns a default value to put in residual parameters, given the index of a
  // custom numeric attribute in the model
  double DefaultResidualSelection(const mjModel* m, int numeric_index);

  // Clamp x between bounds, e.g., bounds[0] <= x[i] <= bounds[1]
  void Clamp(double* x, const double* bounds, int n);

  // get sensor data using string
  double* SensorByName(const mjModel* m, const mjData* d, const std::string& name);

  double DefaultParameterValue(const mjModel* model, std::string_view name);

  int ParameterIndex(const mjModel* model, std::string_view name);

  int CostTermByName(const mjModel* m, const std::string& name);

  // return total size of sensors of type user
  int ResidualSize(const mjModel* model);

  // sanity check that residual size equals total user-sensor dimension
  void CheckSensorDim(const mjModel* model, int residual_size);

  // get traces from sensors
  void GetTraces(double* traces, const mjModel* m, const mjData* d, int num_trace);

  // get keyframe `qpos` data using string
  double* KeyQPosByName(const mjModel* m, const mjData* d, const std::string& name);

  // fills t with N numbers, starting from t0 and incrementing by t_step
  void LinearRange(double* t, double t_step, double t0, int N);

  // find interval in monotonic sequence containing value
  template <typename T>
  void FindInterval(int* bounds, const std::vector<T>& sequence, double value, int length)
  {
    // get bounds
    auto it = std::upper_bound(sequence.begin(), sequence.begin() + length, value);
    int upper_bound = it - sequence.begin();
    int lower_bound = upper_bound - 1;

    // set bounds
    if (lower_bound < 0)
    {
      bounds[0] = 0;
      bounds[1] = 0;
    }
    else if (lower_bound > length - 1)
    {
      bounds[0] = length - 1;
      bounds[1] = length - 1;
    }
    else
    {
      bounds[0] = mju_max(lower_bound, 0);
      bounds[1] = mju_min(upper_bound, length - 1);
    }
  }

  // zero-order interpolation
  void ZeroInterpolation(double* output, double x, const std::vector<double>& xs, const double* ys, int dim,
                         int length, const std::vector<int>& indices = {});

  // linear interpolation
  void LinearInterpolation(double* output, double x, const std::vector<double>& xs, const double* ys, int dim,
                           int length, const std::vector<int>& indices = {});

  // coefficients for cubic interpolation
  void CubicCoefficients(double* coefficients, double x, const std::vector<double>& xs, int T);

  // finite-difference vector
  double FiniteDifferenceSlope(double x, const std::vector<double>& xs, const double* ys, int dim, int length,
                               int i);

  // cubic polynomial interpolation
  void CubicInterpolation(double* output, double x, const std::vector<double>& xs, const double* ys, int dim,
                          int length, const std::vector<int>& indices = {});

  // returns the path to the directory containing the current executable
  std::string GetExecutableDir();

  // returns path to a model XML file given path relative to models dir
  std::string GetModelPath(std::string_view path);

  // dx = (x2 - x1) / h
  void Diff(mjtNum* dx, const mjtNum* x1, const mjtNum* x2, mjtNum h, int n);

  // finite-difference two state vectors ds = (s2 - s1) / h
  void StateDiff(const mjModel* m, mjtNum* ds, const mjtNum* s1, const mjtNum* s2, mjtNum h);

  // return global height of nearest geom in geomgroup under given position
  mjtNum Ground(const mjModel* model, const mjData* data, const mjtNum pos[3],
                const mjtByte* geomgroup = nullptr);

  // set x to be the point on the segment [p0 p1] that is nearest to x
  void ProjectToSegment(double x[3], const double p0[3], const double p1[3]);

  // find frame that best matches 4 feet, z points to body
  void FootFrame(double feet_pos[3], double feet_mat[9], double feet_quat[4], const double body[3],
                 const double foot0[3], const double foot1[3], const double foot2[3], const double foot3[3]);

  // default cost colors
  extern const float CostColors[20][3];
  constexpr int kNCostColors = sizeof(CostColors) / (sizeof(float) * 3);

  // plots - vertical line
  void PlotVertical(mjvFigure* fig, double time, double min_value, double max_value, int N, int index);

  // plots - update data
  void PlotUpdateData(mjvFigure* fig, double* bounds, double x, double y, int length, int index, int x_update,
                      int y_update, double x_bound_lower);

  // plots - reset
  void PlotResetData(mjvFigure* fig, int length, int index);

  // plots - horizontal line
  void PlotHorizontal(mjvFigure* fig, const double* xs, double y, int length, int index);

  // plots - set data
  void PlotData(mjvFigure* fig, double* bounds, const double* xs, const double* ys, int dim, int dim_limit,
                int length, int start_index, double x_bound_lower);

  // add geom to scene
  void AddGeom(mjvScene* scene, mjtGeom type, const mjtNum size[3], const mjtNum pos[3], const mjtNum mat[9],
               const float rgba[4]);

  // add connector geom to scene
  void AddConnector(mjvScene* scene, mjtGeom type, mjtNum width, const mjtNum from[3], const mjtNum to[3],
                    const float rgba[4]);

  // number of available hardware threads
  int NumAvailableHardwareThreads();

  // check mjData for warnings, return true if any warnings
  bool CheckWarnings(mjData* data);

  // compute vector with log-based scaling between min and max values
  void LogScale(double* values, double max_value, double min_value, int steps);

  // get a pointer to a specific element of a vector, or nullptr if out of bounds
  template <typename T>
  inline T* DataAt(std::vector<T>& vec, typename std::vector<T>::size_type elem)
  {
    if (elem < vec.size())
    {
      return &vec[elem];
    }
    else
    {
      return nullptr;
    }
  }

  // increases the value of an atomic variable.
  // in C++20 atomic::operator+= is built-in for floating point numbers, but this
  // function works in C++11
  inline void IncrementAtomic(std::atomic<double>& v, double a)
  {
    for (double t = v.load(); !v.compare_exchange_weak(t, t + a);)
    {
    }
  }

  // get a pointer to a specific element of a vector, or nullptr if out of bounds
  template <typename T>
  inline const T* DataAt(const std::vector<T>& vec, typename std::vector<T>::size_type elem)
  {
    return DataAt(const_cast<std::vector<T>&>(vec), elem);
  }

  using UniqueMjData = std::unique_ptr<mjData, void (*)(mjData*)>;

  inline UniqueMjData MakeUniqueMjData(mjData* d) { return UniqueMjData(d, mj_deleteData); }

  using UniqueMjModel = std::unique_ptr<mjModel, void (*)(mjModel*)>;

  inline UniqueMjModel MakeUniqueMjModel(mjModel* d) { return UniqueMjModel(d, mj_deleteModel); }

  // returns point in 2D convex hull that is nearest to query
  void NearestInHull(mjtNum res[2], const mjtNum query[2], const mjtNum* points, const int* hull, int num_hull);

  // find the convex hull of a set of 2D points
  int Hull2D(int* hull, int num_points, const mjtNum* points);

  // TODO(etom): move findiff-related functions to a different library.

  // finite-difference gradient
  class FiniteDifferenceGradient
  {
  public:
    // constructor
    explicit FiniteDifferenceGradient(int dim);

    // resize memory
    void Resize(int dim);

    // compute gradient
    void Compute(std::function<double(const double* x)> func, const double* input, int dim);

    // members
    std::vector<double> gradient;
    double epsilon = 1.0e-5;

  private:
    std::vector<double> workspace_;
  };

  // finite-difference Jacobian
  class FiniteDifferenceJacobian
  {
  public:
    // constructor
    FiniteDifferenceJacobian(int num_output, int num_input);

    // resize memory
    void Resize(int num_output, int num_input);

    // compute Jacobian
    void Compute(std::function<void(double* output, const double* input)> func, const double* input,
                 int num_output, int num_input);

    // members
    std::vector<double> jacobian;
    std::vector<double> jacobian_transpose;
    std::vector<double> output;
    std::vector<double> output_nominal;
    double epsilon = 1.0e-5;

  private:
    std::vector<double> workspace_;
  };

  // finite-difference Hessian
  class FiniteDifferenceHessian
  {
  public:
    // constructor
    explicit FiniteDifferenceHessian(int dim);

    // resize memory
    void Resize(int dim);

    // compute
    void Compute(std::function<double(const double* x)> func, const double* input, int dim);

    // members
    std::vector<double> hessian;
    double epsilon = 1.0e-5;

  private:
    std::vector<double> workspace1_;
    std::vector<double> workspace2_;
    std::vector<double> workspace3_;
  };

  // set scaled block (size: rb x cb) in mat (size: rm x cm) given mat upper row
  // and left column indices (ri, ci)
  void SetBlockInMatrix(double* mat, const double* block, double scale, int rm, int cm, int rb, int cb, int ri,
                        int ci);

  // set scaled block (size: rb x cb) in mat (size: rm x cm) given mat upper row
  // and left column indices (ri, ci)
  void AddBlockInMatrix(double* mat, const double* block, double scale, int rm, int cm, int rb, int cb, int ri,
                        int ci);

  // get block (size: rb x cb) from mat (size: rm x cm) given mat upper row
  // and left column indices (ri, ci)
  void BlockFromMatrix(double* block, const double* mat, int rb, int cb, int rm, int cm, int ri, int ci);

  // differentiate mju_subQuat wrt qa, qb
  void DifferentiateSubQuat(double jaca[9], double jacb[9], const double qa[4], const double qb[4]);

  // differentiate velocity by finite-differencing two positions wrt to qpos1,
  // qpos2
  void DifferentiateDifferentiatePos(double* jac1, double* jac2, const mjModel* model, double dt,
                                     const double* qpos1, const double* qpos2);

  // compute number of nonzeros in band matrix
  int BandMatrixNonZeros(int ntotal, int nband);

  // TODO(etom): rename (SecondsSince?)
  double GetDuration(std::chrono::steady_clock::time_point time);

  // copy symmetric band matrix block by block
  void SymmetricBandMatrixCopy(double* res, const double* mat, int dblock, int nblock, int ntotal,
                               int num_blocks, int res_start_row, int res_start_col, int mat_start_row,
                               int mat_start_col, double* scratch);

  // zero block (size: rb x cb) in mat (size: rm x cm) given mat upper row
  // and left column indices (ri, ci)
  void ZeroBlockInMatrix(double* mat, int rm, int cm, int rb, int cb, int ri, int ci);

  // square dense to block band matrix
  void DenseToBlockBand(double* res, int dim, int dblock, int nblock);

  // infinity norm
  template <typename T>
  T InfinityNorm(T* x, int n)
  {
    return std::abs(*std::max_element(x, x + n, [](T a, T b) -> bool { return (std::abs(a) < std::abs(b)); }));
  }

  // trace of square matrix
  double Trace(const double* mat, int n);

  // determinant of 3x3 matrix
  double Determinant3(const double* mat);

  // inverse of 3x3 matrix
  void Inverse3(double* res, const double* mat);

  // condition matrix: res = mat11 - mat10 * mat00 \ mat10^T; return rank of mat00
  // TODO(taylor): thread
  void ConditionMatrix(double* res, const double* mat, double* mat00, double* mat10, double* mat11,
                       double* tmp0, double* tmp1, int n, int n0, int n1, double* bandfactor = NULL,
                       int nband = 0);

  // principal eigenvector of 4x4 matrix
  // QUEST algorithm from "Three-Axis Attitude Determination from Vector
  // Observations"
  void PrincipalEigenVector4(double* res, const double* mat, double eigenvalue_init = 12.0);

  // set scaled symmetric block matrix in band matrix
  void SetBlockInBand(double* band, const double* block, double scale, int ntotal, int nband, int nblock,
                      int shift, int row_skip = 0, bool add = true);
} // namespace mjpc

#endif  // MJPC_UTILITIES_H_
