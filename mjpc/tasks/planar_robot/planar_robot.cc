#include "mjpc/tasks/planar_robot/planar_robot.h"

#include <absl/random/random.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>

#include <string>

#include "mjpc/planners/planner.h"
#include "mjpc/planners/rmp/include/util/rmp_util.h"
#include "mjpc/utilities.h"

namespace mjpc {
// task_panda_bring.xml
// task_panda_robotiq_bring.xml
std::string PlanarRobot::XmlPath() const { return GetModelPath("planar_robot/task_timevarying.xml"); }
std::string PlanarRobot::Name() const { return "PlanarRobot"; }

std::string PlanarRobot::URDFPath() const { return mjpc::GetModelPath("planar_robot/planar_2dof.urdf"); }

void PlanarRobot::ResidualFn::Residual(const mjModel* model, const mjData* data, double* residual) const {}

void PlanarRobot::TransitionLocked(mjModel* model, mjData* data) { Task::TransitionLocked(model, data); }

void PlanarRobot::ResetLocked(const mjModel* model) {}

void PlanarRobot::QueryObstacleStatesX() {
  MJPC_LOCK_TASK_DATA_ACCESS;
  obstacle_statesX_.clear();
  for (auto i = 0; i < GetTotalObstaclesNum(); ++i) {
    std::ostringstream obstacle_name;
    obstacle_name << "obstacle_" << i;
    const std::string obs_name = obstacle_name.str();
    auto obstacle_i_id = QueryBodyId(obs_name.c_str());
    auto obstacle_geom_i_id = QueryGeomId(obs_name.c_str());
    mjtNum* obstacle_mocap_i_pos = QueryBodyMocapPos(obs_name.c_str());

    mjtNum* obstacle_i_size = &model_->geom_size[3 * obstacle_geom_i_id];
    // mju_scl(obstacle_size, obstacle_size, 2.0, 3);
    mjtNum* obstacle_i_pos = &data_->xpos[3 * obstacle_i_id];
    // mjtNum* obstacle_i_rot_mat = &data_->geom_xmat[9 * obstacle_geom_i_id];
    mjtNum* obstacle_i_rot = &data_->xquat[4 * obstacle_i_id];

    static constexpr int LIN_IDX = 3;
#if 0
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
    obstacle_statesX_.push_back(
        StateX{.pos_ = rmp::vectorFromScalarArray<StateX::dim>(obstacle_i_pos),
               .rot_ = rmp::quatFromScalarArray<StateX::dim>(obstacle_i_rot).toRotationMatrix(),
               .vel_ = rmp::vectorFromScalarArray<StateX::dim>(obstacle_i_lin_vel),
               .acc_ = rmp::vectorFromScalarArray<StateX::dim>(obstacle_i_lin_acc),
               .size_ = rmp::vectorFromScalarArray<StateX::dim>(obstacle_i_size)});
  }
}

std::vector<double> PlanarRobot::QueryJointPos(int dof) const {
  if (model_ && data_) {
    std::vector<double> qpos(dof, 0);
    memcpy(qpos.data(), data_->qpos + model_->jnt_qposadr[mj_name2id(model_, mjOBJ_JOINT, "joint1")],
           std::min(model_->nq, dof) * sizeof(double));
    return qpos;
  }
  return {};
}

std::vector<double> PlanarRobot::QueryJointVel(int dof) const {
  if (model_ && data_) {
    std::vector<double> qvel(dof, 0);
    memcpy(qvel.data(), data_->qvel + model_->jnt_dofadr[mj_name2id(model_, mjOBJ_JOINT, "joint1")],
           std::min(model_->nv, dof) * sizeof(double));
    return qvel;
  }
  return {};
}

}  // namespace mjpc
