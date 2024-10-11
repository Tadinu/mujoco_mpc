// Copyright 2024 DeepMind Technologies Limited
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

#include "mjpc/tasks/allegro/allegro.h"

#include <string>

// mujoco
#include <mujoco/mujoco.h>

// drake
#include <drake/geometry/proximity_properties.h>
#include <drake/multibody/parsing/parser.h>

// mjpc
#include "mjpc/utilities.h"

#define ALLEGRO_UPSIDE_DOWN (0)  // "whether to treat the hand as upside down (by reversing gravity)"

namespace mjpc {
std::string Allegro::XmlPath() const {
  return GetModelPath(is_target_ball_ ? "allegro/task_ball.xml" : "allegro/task.xml");
}
std::string Allegro::Name() const { return "Allegro"; }

// ------- Residuals for target manipulation task ------
//     Cube position: (3)
//     Cube orientation: (3)
//     Cube linear velocity: (3)
//     Control: (16), there are 16 servos
//     Nominal pose: (16)
//     Joint velocity: (16)
// ------------------------------------------
void Allegro::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
  int counter = 0;
  const auto target_prefix = dynamic_cast<const Allegro *>(task_)->target_type_name();

  // ---------- Cube position ----------
  double *target_position = SensorByName(model, data, target_prefix + "_position");
  double *target_goal_position = SensorByName(model, data, target_prefix + "_goal_position");

  mju_sub3(residual + counter, target_position, target_goal_position);
  counter += 3;

  // ---------- Cube orientation ----------
  double *target_orientation = SensorByName(model, data, target_prefix + "_orientation");
  double *goal_target_orientation = SensorByName(model, data, target_prefix + "_goal_orientation");
  mju_normalize4(goal_target_orientation);

  mju_subQuat(residual + counter, goal_target_orientation, target_orientation);
  counter += 3;

  // ---------- Cube linear velocity ----------
  double *target_linear_velocity = SensorByName(model, data, target_prefix + "_linear_velocity");

  mju_copy(residual + counter, target_linear_velocity, 3);
  counter += 3;

  // ---------- Control ----------
  mju_copy(residual + counter, data->actuator_force, model->nu);
  counter += model->nu;

  // ---------- Nominal Pose ----------
  mju_sub(residual + counter, data->qpos + 7, model->key_qpos + 7, 16);
  counter += 16;

  // ---------- Joint Velocity ----------
  mju_copy(residual + counter, data->qvel + 6, 16);
  counter += 16;

  // Sanity check
  CheckSensorDim(model, counter);
}

void Allegro::TransitionLocked(mjModel *model, mjData *data) {
  // Check for contact between the target and the floor
  int target_geom = mj_name2id(model, mjOBJ_GEOM, target_geom_name().c_str());
  int floor = mj_name2id(model, mjOBJ_GEOM, "floor");

  bool on_floor = false;
  for (int i = 0; i < data->ncon; i++) {
    mjContact *g = data->contact + i;
    if ((g->geom1 == target_geom && g->geom2 == floor) || (g->geom2 == target_geom && g->geom1 == floor)) {
      on_floor = true;
      break;
    }
  }

  // If the target is on the floor and not moving, reset it
  double *target_lin_vel = SensorByName(model, data, target_type_name() + "_linear_velocity");
  if (on_floor && (is_target_ball_ || mju_norm3(target_lin_vel) < 0.001)) {
    int target_body = mj_name2id(model, mjOBJ_BODY, target_body_name().c_str());
    if (target_body != -1) {
      int jnt_qposadr = model->jnt_qposadr[model->body_jntadr[target_body]];
      int jnt_veladr = model->jnt_dofadr[model->body_jntadr[target_body]];
      mju_copy(data->qpos + jnt_qposadr, model->qpos0 + jnt_qposadr, 7);
      mju_zero(data->qvel + jnt_veladr, 6);
    }

    // Step the simulation forward
    mutex_.unlock();
    mj_forward(model, data);
    mutex_.lock();
  }
}

// ===========================================================================================================
// DRAKE IMPL --
//
using drake::geometry::AddCompliantHydroelasticProperties;
using drake::geometry::AddContactMaterial;
using drake::geometry::Box;
using drake::geometry::Cylinder;
using drake::geometry::ProximityProperties;
using drake::geometry::Rgba;
using drake::geometry::Sphere;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::math::RotationMatrixd;
using drake::multibody::CoulombFriction;
using drake::multibody::ModelInstanceIndex;
using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::multibody::RigidBody;
using drake::multibody::SpatialInertia;
using drake::multibody::UnitInertia;
using Eigen::Vector3d;

void Allegro::InitMeshcat() {
  using drake::geometry::Cylinder;
  using drake::geometry::Rgba;
  // Set the camera viewpoint
  const Vector3d camera_pose(0.3, 0.0, 0.5);
  const Vector3d target_pose(0.0, 0.0, 0.0);
  meshcat_->SetCameraPose(camera_pose, target_pose);

  // Add a visualization of the desired ball pose
  const double basis_length = 0.1;
  const double basis_radius = 0.005;
  const double opacity = 0.3;
  meshcat_->SetObject("/desired_pose/x_basis", Cylinder(basis_radius, basis_length),
                      Rgba(1.0, 0.0, 0.0, opacity));
  meshcat_->SetObject("/desired_pose/y_basis", Cylinder(basis_radius, basis_length),
                      Rgba(0.0, 1.0, 0.0, opacity));
  meshcat_->SetObject("/desired_pose/z_basis", Cylinder(basis_radius, basis_length),
                      Rgba(0.0, 0.0, 1.0, opacity));

  const RigidTransformd Xx(RollPitchYawd(0, M_PI_2, 0), Vector3d(basis_length / 2, 0, 0));
  const RigidTransformd Xy(RollPitchYawd(M_PI_2, 0, 0), Vector3d(0, basis_length / 2, 0));
  const RigidTransformd Xz(Vector3d(0, 0, basis_length / 2));
  meshcat_->SetTransform("/desired_pose/x_basis", Xx);
  meshcat_->SetTransform("/desired_pose/y_basis", Xy);
  meshcat_->SetTransform("/desired_pose/z_basis", Xz);
}

void Allegro::UpdateMeshcatFromIdtoConfigs() {
  // Visualize the target pose for the ball
  const Eigen::Vector3d target_position = idto_configs_->q_nom_end.tail(3);
  const RotationMatrixd target_orientation(
      drake::Quaternion<double>(idto_configs_->q_nom_end[16], idto_configs_->q_nom_end[17],
                                idto_configs_->q_nom_end[18], idto_configs_->q_nom_end[19]));

  const RigidTransformd X_desired(target_orientation, target_position);
  meshcat_->SetTransform("/desired_pose", X_desired);
}

void Allegro::CreateDrakePlantModel(drake::multibody::MultibodyPlant<double> *plant) const {
  const drake::Vector4<double> blue(0.2, 0.3, 0.6, 1.0);
  const drake::Vector4<double> black(0.0, 0.0, 0.0, 1.0);

  // Add a model of the hand
  std::string sdf_file = std::filesystem::path(XmlPath()).parent_path() / "allegro_hand_right.sdf";
  Parser(plant).AddModels(sdf_file);
  RigidTransformd X_hand(RollPitchYawd(0, -M_PI_2, 0), Vector3d::Zero());
  plant->WeldFrames(plant->world_frame(), plant->GetFrameByName("hand_root"), X_hand);

  // Define gravity (so we can turn the hand upside down)
  if constexpr (ALLEGRO_UPSIDE_DOWN) {
    plant->mutable_gravity_field().set_gravity_vector(Vector3d(0, 0, 9.81));
  }

  // Add a free-floating ball
  const auto target_name = target_body_name();
  const auto target_id = QueryBodyId(target_name.c_str());
  const double mass = QueryBodyMass(target_name.c_str());
  const double radius = QueryGeomSize(target_geom_name().c_str())[0];
  const mjtNum *position = QueryBodyPos(target_id);
  const auto target_pose = RigidTransformd(Vector3d(position[0], position[1], position[2]));

  ModelInstanceIndex ball_idx = plant->AddModelInstance(target_name);
  const SpatialInertia<double> I(mass, Vector3d::Zero(), UnitInertia<double>::SolidSphere(radius));
  const RigidBody<double> &target = plant->AddRigidBody(target_name, ball_idx, I);
  plant->RegisterVisualGeometry(target, RigidTransformd::Identity(), Sphere(radius), target_name + "_visual",
                                blue);
  ProximityProperties target_proximity;
  AddContactMaterial(3.0, {}, CoulombFriction<double>(1.0, 1.0), &target_proximity);
  AddCompliantHydroelasticProperties(0.01, 5e5, &target_proximity);
  plant->RegisterCollisionGeometry(target, target_pose, Sphere(radius), target_name + "_collision",
                                   target_proximity);

  // Add some markers to the ball so we can see its rotation
  RigidTransformd X_m1(RollPitchYawd(0, 0, 0), Vector3d(0, 0, 0));
  RigidTransformd X_m2(RollPitchYawd(M_PI_2, 0, 0), Vector3d(0, 0, 0));
  RigidTransformd X_m3(RollPitchYawd(0, M_PI_2, 0), Vector3d(0, 0, 0));
  plant->RegisterVisualGeometry(target, X_m1, Cylinder(0.1 * radius, 2 * radius), target_name + "_marker_one",
                                black);
  plant->RegisterVisualGeometry(target, X_m2, Cylinder(0.1 * radius, 2 * radius), target_name + "_marker_two",
                                black);
  plant->RegisterVisualGeometry(target, X_m3, Cylinder(0.1 * radius, 2 * radius),
                                target_name + "_marker_three", black);

  // Add some markers to show the ball's orientation with the same colors as
  // the target frame
  const RigidTransformd Xx(RollPitchYawd(0, M_PI_2, 0), Vector3d(radius / 2, 0, 0));
  const RigidTransformd Xy(RollPitchYawd(M_PI_2, 0, 0), Vector3d(0, radius / 2, 0));
  const RigidTransformd Xz(Vector3d(0, 0, radius / 2));
  plant->RegisterVisualGeometry(target, Xx, Cylinder(0.1 * radius, radius * 1.01), target_name + "_axis_x",
                                drake::Vector4<double>(1.0, 0.0, 0.0, 1.0));
  plant->RegisterVisualGeometry(target, Xy, Cylinder(0.1 * radius, radius * 1.01), target_name + "_axis_y",
                                drake::Vector4<double>(0.0, 1.0, 0.0, 1.0));
  plant->RegisterVisualGeometry(target, Xz, Cylinder(0.1 * radius, radius * 1.01), target_name + "_axis_z",
                                drake::Vector4<double>(0.0, 0.0, 1.0, 1.0));

  // Add the ground, slightly below the allegro hand
  const mjtNum *ground_position = QueryGeomPos("floor");
  const auto ground_pose =
      RigidTransformd(Vector3d(ground_position[0], ground_position[1], ground_position[2]));
  plant->RegisterCollisionGeometry(plant->world_body(), ground_pose, Box(25, 25, 10), "ground",
                                   CoulombFriction<double>(1.0, 1.0));
}
}  // namespace mjpc
