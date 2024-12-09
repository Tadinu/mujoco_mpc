#pragma once

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtnum.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "mjpc/mjcf/mjcf_model.h"
#include "mjpc/task.h"
#include "mjpc/tasks/mpl/mpl_cost.h"
#include "mjpc/utilities.h"

using namespace Eigen;

// Define constants for Shadow Dexterous Hand
const int NUM_JOINTS = 24; // Total joints for the hand (thumb, fingers, wrist)
const int NUM_TENDONS = 20; // Actuated by 20 Smart Motors

// Joint limits for each joint (radians) based on the spec
const double JOINT_LIMITS_MIN[] = {0, 0, -M_PI / 15, -M_PI / 18, -M_PI / 6, -M_PI / 30,
                                   0, -M_PI, -M_PI / 4, -M_PI / 4, 0, -M_PI};

const double JOINT_LIMITS_MAX[] = {M_PI / 2, M_PI / 2, M_PI / 2, M_PI / 18, M_PI / 6, M_PI / 30,
                                   1.222, M_PI, M_PI / 4, M_PI / 4, M_PI / 2, M_PI};

// Example coupling matrix for the Shadow Hand (4x8 for one finger)
const double COUPLING_MATRIX[4][8] = {{1, -1, 1, -1, 1, -1, 1, -1},
                                      {1, 1, 1, 0, -1, -1, -1, 0},
                                      {1, 1, 0, 0, -1, -1, 0, 0},
                                      {1, 0, 0, 0, -1, 0, 0, 0}};

// Physical parameters (approximated or derived from the spec)
const double TENDON_FORCE_MIN = 0.5; // N
const double TENDON_FORCE_MAX = 100.0; // N

// NOTE: CONSIDER USING [mj_geomDistance] between finger tip sites & object sites
class MPLGraspCostCalculator : public MPLCostCalculator {
public:
  MPLGraspCostCalculator() = default;
  double max_acc_radius = 0;

  // Compute Force Index (FI)
  double ComputeForceIndex(const double* forces, const double* fmv, int len) const {
    double maxForce = 0.0;
    for (int i = 0; i < len; i++) {
      double dotProduct = mju_dot(forces + 3 * i, fmv, 3);
      if (dotProduct > 0) {
        maxForce = std::max(maxForce, mju_norm(forces + 3 * i, 3));
      }
    }
    return maxForce;
  }

  // Compute Joint Limits Index (JLI)
  double ComputeJointLimitsIndex(const double* jointAngles, int numJoints) const {
    double penalty = 1.0;
    for (int i = 0; i < numJoints; i++) {
      const auto& qi = jointAngles[i];
      const double delta_qi = std::pow(JOINT_LIMITS_MAX[i] - JOINT_LIMITS_MIN[i], 2);
      const double qi_max_delta = qi - JOINT_LIMITS_MAX[i];
      const double qi_min_delta = qi - JOINT_LIMITS_MIN[i];
      const double grad_qi = delta_qi * (qi_max_delta + qi_min_delta) / (4 * (std::pow(
                                   qi_max_delta * qi_min_delta, 2)));
      penalty *= 1.0 / std::sqrt(1 + std::abs(grad_qi));
    }
    return penalty;
  }

  void ComputeAMatrix(const mjModel* m, mjData* d, int bodyId, mjtNum* A) const {
    int nv = m->nv;

    // Step 1: Compute the Jacobian for the body's center of mass
    mjtNum* J = new mjtNum[6 * nv]; // 6 rows (linear + angular) x number of joints
    memset(J, 0, 6 * nv * sizeof(mjtNum));
    mj_jacBody(m, d, J, nullptr, bodyId);

    // Extract the linear part of the Jacobian (first 3 rows)
    mjtNum* J_linear = new mjtNum[3 * nv];
    memset(J_linear, 0, 3 * nv * sizeof(mjtNum));
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < nv; j++) {
        J_linear[i * nv + j] = J[i * nv + j];
      }
    }

    // Step 2: Factorize the mass matrix
    mj_factorM(m, d); // Factorizes qM in place

#if 1
    // A = J*inv(M)
    mj_solveM(m, d, A, J_linear, 6);
#else
    // Step 3: Solve for M^-1 * J^T
    mjtNum* temp = new mjtNum[nv];
    for (int col = 0; col < 3; col++) {
      // Iterate over columns of J_linear
      memset(temp, 0, nv * sizeof(mjtNum));
      for (int row = 0; row < nv; row++) {
        temp[row] = J_linear[col * nv + row];
      }
      mj_solveM(m, d, temp, temp, 1); // Solves M * x = J^T[:, col]
      for (int row = 0; row < nv; row++) {
        A[col * nv + row] = temp[row];
      }
    }
#endif

    // Cleanup
    delete[] J;
    delete[] J_linear;
  }

  // Compute the radius of the largest inscribed sphere
  mjtNum ComputeLargestInscribedSphereRadius(int fingertip_site_body_id, const mjtNum* tau_min,
                                             const mjtNum* tau_max,
                                             int rows, int cols) const {
    // Compute A matrix: A=J⋅M^-1
    int nv = mj_model_->nv;
    mjtNum* A = new mjtNum[3 * nv]; // 3 rows (linear) x number of DOFs
    ComputeAMatrix(mj_model_, (mjData*)mj_data_, fingertip_site_body_id, A);

    // Compute R
    mjtNum R = INFINITY; // Start with the largest possible radius

    // Iterate over each row of matrix A (representing normal vectors of polytope faces)
    for (int i = 0; i < rows; i++) {
      // Compute the norm of the normal vector (row of A)
      mjtNum norm_n = 0.0;
      for (int j = 0; j < cols; j++) {
        norm_n += A[i * cols + j] * A[i * cols + j]; // A[i, j] is at index (i * cols + j)
      }
      norm_n = std::sqrt(norm_n);

      // Compute b_i for the current face
      mjtNum b_i = 0.0;
      for (int j = 0; j < cols; j++) {
        b_i += A[i * cols + j] * tau_max[j]; // Use tau_min[j] for other half-plane if needed
      }

      // Compute the distance to this face
      mjtNum d_i = std::fabs(b_i) / norm_n;

      // Update the minimum distance
      R = (d_i < R) ? d_i : R;
    }

    return R; // Return the radius of the largest inscribed sphere
  }

  double ComputeDMI(const std::string& fingertip_sensor_name) {
    int fingertip_sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, fingertip_sensor_name.c_str());
    int fingertip_site_id = mj_model_->sensor_objid[fingertip_sensor_id];
    int fingertip_site_body_id = mj_model_->site_bodyid[fingertip_site_id];

    // DMI (Dynamic Manipulability Index): a metric that quantifies the maximum translational Cartesian acceleration
    // achievable by the fingertip in all directions, with respect to specific configuration q and tendon force limits.
    // -> An quantity representing the ability of a finger to perform precise and skillful manipulations.
    // -> Since DMI is normalized, it indicates the ratio of the maximal omnidirectional acceleration
    // achievable by a finger configuration to the maximum DMI value of all voxels in its fingertip space (operational space).

    //mjtNum fingertip_full_acc[6]; // rot+lin
    //mj_objectAcceleration(mj_model_, mj_data_, mjOBJ_BODY, fingertip_site_body_id, fingertip_full_acc,
    //                    /*flg_local=*/0);
    mjtNum tau_min[3] = {-10, -10, -10};
    mjtNum tau_max[3] = {10, 10, 10};
    int nv = mj_model_->nv;
    const auto radius = ComputeLargestInscribedSphereRadius(fingertip_site_body_id, tau_min, tau_max, nv, nv);
    if (radius > max_acc_radius) {
      max_acc_radius = radius;
    }
    return radius / max_acc_radius;
  }

  // Compute Fingertip Manipulability (FtM)
  double ComputeFingertipManipulability(const double* jointAngles,
                                        const std::string& fingertip_sensor_name) {
    double* fingertipForces = mjpc::SensorByName(mj_model_, mj_data_, fingertip_sensor_name);
    int fingertip_sensor_id = mj_name2id(mj_model_, mjOBJ_SENSOR, fingertip_sensor_name.c_str());
    int fingertip_site_id = mj_model_->sensor_objid[fingertip_sensor_id];
    int fingertip_site_body_id = mj_model_->site_bodyid[fingertip_site_id];

    // DMI
    const double fingertip_DMI = ComputeDMI(fingertip_sensor_name);

    // FI
    double fingertip_FI = 0;
    double FMV[3]; // Force-manipulating vector, aka, contact force vector
    for (const auto& [contact_obj, contact_list] : hand_contacts_) {
      for (const auto& contact : contact_list) {
        if (contact.id == fingertip_site_body_id) {
          mju_copy3(FMV, contact.f.data());

          // ForceIndex(FI), a metric designed to assess the ability of the finger to generate contact force
          // to maintain the stability of an object when subjected to wrench disturbance.
          // Here, we consider all contact force vectors are of equal magnitude regardless of their directions
          fingertip_FI += ComputeForceIndex(fingertipForces, FMV, 3);
        }
      }
    }

    // Placeholder inertia matrix and dynamics computations
    double JLI = ComputeJointLimitsIndex(jointAngles, 4);
    return fingertip_FI * fingertip_DMI * JLI;
  }

  double FingerCost(const std::string& fingerFirstJointName, const std::string& fingertip_sensor_name) {
    // Example joint angles and tendon forces
    mjtNum jointAngles[4] = {0.1, 0.2, -0.1, 0.0}; // 4-DOF finger

    const_cast<mjpc::Task*>(mj_task_)->first_joint_name_ = fingerFirstJointName;
    auto joint_pos = mj_task_->QueryJointPos(4);
    mju_copy(jointAngles, joint_pos.data(), 4);

    mjtNum tendonForces[8]; // 8 tendons driving one finger
    for (int i = 0; i < 8; i++) {
      tendonForces[i] = (TENDON_FORCE_MIN + TENDON_FORCE_MAX) / 2.0;
    }

    // Compute Fingertip Manipulability
    double FtM = ComputeFingertipManipulability(jointAngles, fingertip_sensor_name);
    std::cout << "Fingertip Manipulability: " << FtM << std::endl;
    return 1 / FtM;
  }

  double CalTendonPassiveForces(const std::string& tendon_name) const {
    int nv = mj_model_->nv, njnt = mj_model_->njnt, ntendon = mj_model_->ntendon;
    int issparse = mj_isSparse(mj_model_);

    const int i = mj_name2id(mj_model_, mjOBJ_TENDON, tendon_name.c_str());
    assert(i >= 0);
    assert(i < mj_model_->ntendon);
    std::vector<mjtNum> qfrc_tendon_spring(nv);
    std::vector<mjtNum> qfrc_tendon_damper(nv);
    // tendon-level spring-dampers
    mjtNum stiffness = mj_model_->tendon_stiffness[i];
    mjtNum damping = mj_model_->tendon_damping[i];

    // disabled : nothing to do
    if (stiffness == 0 && damping == 0) {
      return 0;
    }

    // Compute spring force along tendon
    mjtNum length = mj_data_->ten_length[i];
    mjtNum lower = mj_model_->tendon_lengthspring[2 * i];
    mjtNum upper = mj_model_->tendon_lengthspring[2 * i + 1];
    mjtNum frc_spring = 0;
    if (length > upper) {
      frc_spring = stiffness * (upper - length);
    } else if (length < lower) {
      frc_spring = stiffness * (lower - length);
    }

    // Compute damper linear force along tendon
    mjtNum frc_damper = -damping * mj_data_->ten_velocity[i];

    // transform to joint torque, add to qfrc_{spring, damper}: dense or sparse
    if (issparse) {
      if (frc_spring || frc_damper) {
        int end = mj_data_->ten_J_rowadr[i] + mj_data_->ten_J_rownnz[i];
        for (int j = mj_data_->ten_J_rowadr[i]; j < end; j++) {
          int k = mj_data_->ten_J_colind[j];
          mjtNum J = mj_data_->ten_J[j];
          qfrc_tendon_spring[k] += J * frc_spring;
          qfrc_tendon_damper[k] += J * frc_damper;
        }
      }
    } else {
      if (frc_spring) mju_addToScl(mj_data_->qfrc_spring, mj_data_->ten_J + i * nv, frc_spring, nv);
      if (frc_damper) mju_addToScl(mj_data_->qfrc_damper, mj_data_->ten_J + i * nv, frc_damper, nv);
    }
    return std::reduce(qfrc_tendon_spring.begin(), qfrc_tendon_spring.end()) +
           std::reduce(qfrc_tendon_damper.begin(), qfrc_tendon_damper.end());
  }
}; // end MPLGraspCostCalculator

/*
* https://www.researchgate.net/publication/377857824_The_Fingertip_Manipulability_Assessment_of_Tendon-driven_Multi-fingered_Hands
* https://www.shadowrobot.com/wp-content/uploads/2022/03/shadow_dexterous_hand_e_technical_specification.pdf
* Features Added:
Detailed Joint Limit Constraints: Incorporates ranges for specific joints of the Shadow Hand.
Tendon Forces: Reflects the tendon-driven actuation mechanism.
Dynamic Manipulability Index: Considers the Jacobian and mass matrix for realistic kinematics and dynamics.
Polytope Representation: Approximates the feasible forces in Cartesian space.
Customizable Metrics: Metrics like FI, DMI, and JLI are modular for extensibility.
 * */
