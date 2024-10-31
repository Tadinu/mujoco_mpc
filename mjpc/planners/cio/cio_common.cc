#include "mjpc/planners/cio/cio_common.h"

#include "mjpc/planners/cio/cio_util.h"

void CIOPose::add_noise() {
  mjpc_casadi::add_gaussian_noise(trans);
  mjpc_casadi::add_gaussian_noise(quat);
}

void CIOVelocity::add_noise() {
  mjpc_casadi::add_gaussian_noise(linear_vel);
  mjpc_casadi::add_gaussian_noise(angular_vel);
}

void CIOAcceleration::add_noise() {
  mjpc_casadi::add_gaussian_noise(linear_acc);
  mjpc_casadi::add_gaussian_noise(angular_acc);
}

void CIOContact::add_noise() {
  mjpc_casadi::add_gaussian_noise(f);
  mjpc_casadi::add_gaussian_noise(ro);
  CIOUtils::add_gaussian_noise(c);
}

const int CIOObservation::pose_size = CIOPose().size();
const int CIOObservation::vel_size = CIOVelocity().size();
const int CIOObservation::acc_size = CIOAcceleration().size();
const int CIOObservation::pose_vel_size = CIOPose().size() + CIOVelocity().size();
const int CIOObservation::pose_vel_acc_size =
    CIOPose().size() + CIOVelocity().size() + CIOAcceleration().size();
const int CIOObservation::contact_size = CIOContact().size();
void CIOObservation::add_noise() {
  pose.add_noise();
  vel.add_noise();
  acc.add_noise();
  for (auto& contact : contacts) {
    contact.add_noise();
  }
}
