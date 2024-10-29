#include "mjpc/planners/cio/cio_common.h"

#include "mjpc/planners/cio/cio_util.h"

void CIOPose::add_noise() {
  CIOUtils::add_gaussian_noise(trans.x());
  CIOUtils::add_gaussian_noise(trans.y());
  CIOUtils::add_gaussian_noise(trans.z());

  CIOUtils::add_gaussian_noise(quat.x());
  CIOUtils::add_gaussian_noise(quat.y());
  CIOUtils::add_gaussian_noise(quat.z());
  CIOUtils::add_gaussian_noise(quat.w());
}

void CIOVelocity::add_noise() {
  CIOUtils::add_gaussian_noise(linear_vel.x());
  CIOUtils::add_gaussian_noise(linear_vel.y());
  CIOUtils::add_gaussian_noise(linear_vel.z());

  CIOUtils::add_gaussian_noise(angular_vel.x());
  CIOUtils::add_gaussian_noise(angular_vel.y());
  CIOUtils::add_gaussian_noise(angular_vel.z());
}

void CIOAcceleration::add_noise() {
  CIOUtils::add_gaussian_noise(linear_acc.x());
  CIOUtils::add_gaussian_noise(linear_acc.y());
  CIOUtils::add_gaussian_noise(linear_acc.z());

  CIOUtils::add_gaussian_noise(angular_acc.x());
  CIOUtils::add_gaussian_noise(angular_acc.y());
  CIOUtils::add_gaussian_noise(angular_acc.z());
}

void CIOContact::add_noise() {
  CIOUtils::add_gaussian_noise(f.x());
  CIOUtils::add_gaussian_noise(f.y());
  CIOUtils::add_gaussian_noise(f.z());

  CIOUtils::add_gaussian_noise(ro.x());
  CIOUtils::add_gaussian_noise(ro.y());
  CIOUtils::add_gaussian_noise(ro.z());

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
