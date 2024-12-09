#include "mjpc/tasks/allegro_x/allegro_x.h"

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
std::string AllegroX::XmlPath() const {
  return GetModelPath(is_target_ball_ ? "allegro_x/task_ball.xml" : "allegro_x/task.xml");
}

std::string AllegroX::Name() const { return "AllegroX"; }

// ------- Residuals for target manipulation task ------
//     Cube position: (3)
//     Cube orientation: (3)
//     Cube linear velocity: (3)
//     Control: (16), there are 16 servos
//     Nominal pose: (16)
//     Joint velocity: (16)
// ------------------------------------------
void AllegroX::ResidualFn::Residual(const mjModel* model, const mjData* data, double* residual) const {
  int counter = 0;
  const auto target_prefix = dynamic_cast<const AllegroX*>(task_)->target_type_name();
#if 0
  // Contact point
  std::map<std::string, std::vector<mjContact*>> contacts;
  const std::vector<std::string> grasp_points_names = {"ball_pt1", "ball_pt2", "ball_pt3", "ball_pt4"};
  for (const auto& grasp_point_name : grasp_points_names) {
    for (int i = 0; i < data->ncon; ++i) {
      std::cout << i << std::endl;
      const auto& contact_i = data->contact[i];

      // Get contact force/torque, rotate into traj frame, then site frame.
      // Note that contact.frame is column major.
      mjtNum conforce[6], conray[3];
      // get contact force:torque in contact frame
      mj_contactForce(model, data, i, conforce);

      // convert contact normal force to global frame, normalize
      mju_mulMatTVec3(conray, contact_i.frame, conforce);
      mju_normalize3(conray);
      mjpc::print(conray);
    }
  }

  // ---------- Grasp points poses ----------
#endif

#if 1
  static const std::vector<std::string> fingertip_names = {"rf_tip", "mf_tip", "ff_tip", "th_tip"};
  static const std::vector<std::string> grasp_points_names = {"ball_pt1", "ball_pt2", "ball_pt3", "ball_pt4"};
  for (const auto& fingertip_name : fingertip_names) {
    auto* fingertip_pos = task_->QuerySitePos(fingertip_name.c_str());
    mjtNum min_distance = 1000;
    for (const auto& grasp_point_name : grasp_points_names) {
      auto* point_pos_i = SensorByName(model, data, "p_" + grasp_point_name);
      auto* point_quat_i = SensorByName(model, data, "q_" + grasp_point_name);

      mjtNum distance = mju_dist3(fingertip_pos, point_pos_i);
      if (distance < min_distance) {
        min_distance = distance;
      }
    }
    residual[counter++] = min_distance;
  }
#else

  // ---------- Cube position ----------
  double* target_position = SensorByName(model, data, target_prefix + "_position");
  double* target_goal_position = SensorByName(model, data, target_prefix + "_goal_position");

  mju_sub3(residual + counter, target_position, target_goal_position);
  counter += 3;

  // ---------- Cube orientation ----------
  double* target_orientation = SensorByName(model, data, target_prefix + "_orientation");
  double* goal_target_orientation = SensorByName(model, data, target_prefix + "_goal_orientation");
  mju_normalize4(goal_target_orientation);

  mju_subQuat(residual + counter, goal_target_orientation, target_orientation);
  counter += 3;

  // ---------- Cube linear velocity ----------
  double* target_linear_velocity = SensorByName(model, data, target_prefix + "_linear_velocity");

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

  // residual[counter++] = cost_calc_.TotalCost();
#endif

  // Sanity check
  CheckSensorDim(model, counter);
}

void AllegroX::TransitionLocked(mjModel* model, mjData* data) {
  // Re-setup [cost_calc_]
  // residual_.cost_calc_.Setup(model, data, this);
#if 0
  // Move fingers to their closest grasp points
  static const std::vector<std::string> fingertip_names = {"rf_tip", "mf_tip", "ff_tip", "th_tip"};
  static const std::vector<std::string> grasp_points_names = {"ball_pt1", "ball_pt2", "ball_pt3", "ball_pt4"};
  for (const auto& fingertip_name : fingertip_names) {
    auto* fingertip_pos = QueryGeomPos(data, fingertip_name);
    mjtNum min_distance = 1000;
    std::string min_distance_grasp_point_name;

    for (const auto& grasp_point_name : grasp_points_names) {
      auto* point_pos_i = SensorByName(model, data, "p_" + grasp_point_name);
      auto* point_quat_i = SensorByName(model, data, "q_" + grasp_point_name);

      mjtNum distance = mju_dist3(fingertip_pos, point_pos_i);
      if (distance < min_distance) {
        min_distance = distance;
        min_distance_grasp_point_name = grasp_point_name;
      }
    }

    // Move fingertip to grasp point of [min_distance_grasp_point_name]
    auto* grasp_point_pos = SensorByName(model, data, "p_" + min_distance_grasp_point_name);
    auto* grasp_point_quat = SensorByName(model, data, "q_" + min_distance_grasp_point_name);
    SetSitePos(grasp_point_pos);
    SetSiteQuat(grasp_point_quat);
    data->site_pos = grasp_point_pos;
#endif

  // Check for contact between the target and the floor
  int target_geom = mj_name2id(model, mjOBJ_GEOM, target_geom_name().c_str());
  int floor = mj_name2id(model, mjOBJ_GEOM, "floor");

  bool on_floor = false;
  for (int i = 0; i < data->ncon; i++) {
    mjContact* g = data->contact + i;
    if ((g->geom1 == target_geom && g->geom2 == floor) || (g->geom2 == target_geom && g->geom1 == floor)) {
      on_floor = true;
      break;
    }
  }

  // If the target is on the floor and not moving, reset it
  double* target_lin_vel = SensorByName(model, data, target_type_name() + "_linear_velocity");
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

void AllegroX::InitMeshcat() {
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

void AllegroX::UpdateMeshcatFromIdtoConfigs() {
  // Visualize the target pose for the ball
  const Eigen::Vector3d target_position = idto_configs_->q_nom_end.tail(3);
  const RotationMatrixd target_orientation(
      drake::Quaternion<double>(idto_configs_->q_nom_end[16], idto_configs_->q_nom_end[17],
                                idto_configs_->q_nom_end[18], idto_configs_->q_nom_end[19]));

  const RigidTransformd X_desired(target_orientation, target_position);
  meshcat_->SetTransform("/desired_pose", X_desired);
}

void AllegroX::CreateDrakePlantModel(drake::multibody::MultibodyPlant<double>* plant) const {
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
  const mjtNum* position = QueryBodyPos(target_id);
  const auto target_pose = RigidTransformd(Vector3d(position[0], position[1], position[2]));

  ModelInstanceIndex ball_idx = plant->AddModelInstance(target_name);
  const SpatialInertia<double> I(mass, Vector3d::Zero(), UnitInertia<double>::SolidSphere(radius));
  const RigidBody<double>& target = plant->AddRigidBody(target_name, ball_idx, I);
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
  const mjtNum* ground_position = QueryGeomPos("floor");
  const auto ground_pose =
      RigidTransformd(Vector3d(ground_position[0], ground_position[1], ground_position[2]));
  plant->RegisterCollisionGeometry(plant->world_body(), ground_pose, Box(25, 25, 10), "ground",
                                   CoulombFriction<double>(1.0, 1.0));
}
}  // namespace mjpc
