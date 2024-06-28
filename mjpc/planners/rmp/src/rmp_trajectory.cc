#include "mjpc/planners/rmp/include/planner/rmp_trajectory.h"

#include <fstream>
#include <iostream>
#include <random>

/** Formatting of the data struct for data exporting */
template <class TSpace>
std::string rmpcpp::RMPWaypoint<TSpace>::getHeaderFormat() {
  return "x y z vx vy vz ax ay az v_mag ";
}
template std::string
rmpcpp::RMPWaypoint<rmpcpp::Space<3>>::getHeaderFormat();

template <>
std::string rmpcpp::RMPWaypoint<rmpcpp::Space<2>>::getHeaderFormat() {
  return "x y vx vy ax ay v_mag ";
}

template <class TSpace>
std::string rmpcpp::RMPWaypoint<TSpace>::format() const {
  Eigen::IOFormat format(Eigen::FullPrecision, Eigen::DontAlignCols, " ", " ",
                         "", "", " ", "");
  std::stringstream str;
  str << position.format(format) << velocity.format(format)
      << acceleration.format(format) << " " << velocity.norm();
  return str.str();
}
/**************************************************/

template <class TSpace>
void rmpcpp::RMPTrajectory<TSpace>::addPoint(const VectorQ &p, const VectorQ &v,
                                             const VectorQ &a) {
  RMPWaypoint<TSpace> point;
  point.position = p;
  point.velocity = v;
  point.acceleration = a;
  addPoint(std::move(point));
}

template <class TSpace>
void rmpcpp::RMPTrajectory<TSpace>::addPoint(const RMPWaypoint<TSpace>& point)
{
  auto new_point = point;
  addPoint(std::move(new_point));
}

template <class TSpace>
void rmpcpp::RMPTrajectory<TSpace>::addPoint(RMPWaypoint<TSpace>&& point)
{
  point.cumulative_length = getLength() + (current().position - point.position).norm();
  trajectory_data_.push_back(std::move(point));
}

template <class TSpace>
int rmpcpp::RMPTrajectory<TSpace>::getSegmentCount() const {
  return trajectory_data_.size() - 1;  // one point is not a segment yet.
}

template <class TSpace>
int rmpcpp::RMPTrajectory<TSpace>::getWaypointsCount() const {
  return trajectory_data_.size();
}

template <class TSpace>
double rmpcpp::RMPTrajectory<TSpace>::getLength() const {
  return (trajectory_data_.size() > 0 ) ? current().cumulative_length : 0.0;
}

template <class TSpace>
double rmpcpp::RMPTrajectory<TSpace>::getSmoothness() const {
  double smoothness = 0;
  for (size_t i = 2; i < trajectory_data_.size(); i++) {
    auto A = trajectory_data_[i - 2].position;
    auto B = trajectory_data_[i - 1].position;
    auto C = trajectory_data_[i].position;
    A = B - A;
    B = C - B;
    smoothness += 1 - 1 / M_PI * atan2((A.cross(B)).norm(), A.dot(B));
  }

  return smoothness / (trajectory_data_.size() - 2);
}


/** Cross product does not work for 2d vectors. */
template <>
double rmpcpp::RMPTrajectory<rmpcpp::Space<2>>::getSmoothness() const {
  throw std::runtime_error("Not implemented");
}

template <class TSpace>
void rmpcpp::RMPTrajectory<TSpace>::writeToStream(std::ofstream &file) const {
  // write header
  file << "i " << rmpcpp::RMPWaypoint<TSpace>::getHeaderFormat()
       << std::endl;

  // write lines
  for (size_t i = 0; i < trajectory_data_.size(); ++i) {
    file << i << trajectory_data_[i].format() << std::endl;
  }
}

// explicit instantation
template class rmpcpp::RMPTrajectory<rmpcpp::Space<3>>;
//template class rmpcpp::RMPTrajectory<rmpcpp::Space<2>>;
template class rmpcpp::RMPTrajectory<rmpcpp::CylindricalSpace>;

