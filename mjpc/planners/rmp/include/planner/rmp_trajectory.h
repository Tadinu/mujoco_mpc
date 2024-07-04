#ifndef RMP_PLANNER_TRAJECTORY_RMP_H
#define RMP_PLANNER_TRAJECTORY_RMP_H

#include <memory>
#include <vector>
#include <Eigen/Dense>

#include "mjpc/trajectory.h"
#include "mjpc/planners/rmp/include/core/rmp_space.h"

namespace rmp {
/*
 * Struct that holds a discretized and integrated point in a RMP trajectory.
 */
template <class TSpace>
struct RMPWaypoint {
  using VectorQ = Eigen::Matrix<double, TSpace::dim, 1>;
  using MatrixQ = Eigen::Matrix<double, TSpace::dim, TSpace::dim>;
  VectorQ position = VectorQ::Zero();
  MatrixQ rotation = MatrixQ::Identity();
  VectorQ velocity = VectorQ::Zero();
  VectorQ acceleration = VectorQ::Zero();
  double cumulative_length = 0.0; // Cumulative length of this trajectory

  static std::string getHeaderFormat();
  std::string format() const;
};

/*
 * Class that holds a full trajectory (vector of points)
 */
template <class TSpace>
class RMPTrajectory : public mjpc::Trajectory {
  using VectorQ = Eigen::Matrix<double, TSpace::dim, 1>;

public:
  RMPWaypoint<TSpace> start() const {
    return (trajectory_data_.size() > 0) ? trajectory_data_[0] : RMPWaypoint<TSpace>();
  }

  RMPWaypoint<TSpace> current() const {
    return (trajectory_data_.size() > 0) ? trajectory_data_.back() : RMPWaypoint<TSpace>();
  }

  void addWaypoint(const VectorQ& p, const VectorQ& v, const VectorQ& a = VectorQ::Zero());
  void addWaypoint(const RMPWaypoint<TSpace>& point);
  void addWaypoint(RMPWaypoint<TSpace>&& point);

  int getSegmentCount() const {
    return trajectory_data_.size() - 1; // one point is not a segment yet.
  }

  int getWaypointsCount() const {
    return trajectory_data_.size();
  }

  double getLength() const {
    return (trajectory_data_.size() > 0) ? current().cumulative_length : 0.0;
  }

  double getSmoothness() const;

  bool hasCollided() const {
    return collided_;
  }

  void setCollided(bool collided) {
    collided_ = collided;
  }

  inline const RMPWaypoint<TSpace> operator[](int i) const {
    return trajectory_data_[i];
  };

  inline RMPWaypoint<TSpace>& operator[](int i) {
    return trajectory_data_[i];
  };

  void clearData() {
    trajectory_data_.clear();
  }

  bool hasData() const {
    return trajectory_data_.size() > 0;
  }

  void setMaxLength(float max_length) {
    max_length_ = max_length;
  }

  void writeToStream(std::ofstream& file) const;

private:
  std::vector<RMPWaypoint<TSpace>> trajectory_data_;
  bool collided_ = false;
  float max_length_ = 0.;
};
} // namespace rmp

#endif  // RMP_PLANNER_TRAJECTORY_RMP_H
