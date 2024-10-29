#include "mjpc/planners/cio/cio_common.h"

#include "mjpc/planners/cio/cio_util.h"

void CIOPose::add_noise() {
  cio_utils::add_gaussian_noise(trans.x());
  cio_utils::add_gaussian_noise(trans.y());
  cio_utils::add_gaussian_noise(trans.z());

  cio_utils::add_gaussian_noise(quat.x());
  cio_utils::add_gaussian_noise(quat.y());
  cio_utils::add_gaussian_noise(quat.z());
  cio_utils::add_gaussian_noise(quat.w());
}

void CIOVelocity::add_noise() {
  cio_utils::add_gaussian_noise(linear_vel.x());
  cio_utils::add_gaussian_noise(linear_vel.y());
  cio_utils::add_gaussian_noise(linear_vel.z());

  cio_utils::add_gaussian_noise(angular_vel.x());
  cio_utils::add_gaussian_noise(angular_vel.y());
  cio_utils::add_gaussian_noise(angular_vel.z());
}

void CIOAcceleration::add_noise() {
  cio_utils::add_gaussian_noise(linear_acc.x());
  cio_utils::add_gaussian_noise(linear_acc.y());
  cio_utils::add_gaussian_noise(linear_acc.z());

  cio_utils::add_gaussian_noise(angular_acc.x());
  cio_utils::add_gaussian_noise(angular_acc.y());
  cio_utils::add_gaussian_noise(angular_acc.z());
}

void CIOContact::add_noise() {
  cio_utils::add_gaussian_noise(f.x());
  cio_utils::add_gaussian_noise(f.y());
  cio_utils::add_gaussian_noise(f.z());

  cio_utils::add_gaussian_noise(ro.x());
  cio_utils::add_gaussian_noise(ro.y());
  cio_utils::add_gaussian_noise(ro.z());
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
