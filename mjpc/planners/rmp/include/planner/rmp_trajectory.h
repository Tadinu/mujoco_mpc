#ifndef RMPCPP_PLANNER_TRAJECTORY_RMP_H
#define RMPCPP_PLANNER_TRAJECTORY_RMP_H

#include <memory>
#include <vector>
#include <Eigen/Dense>

#include "mjpc/trajectory.h"
#include "mjpc/planners/rmp/include/core/rmp_space.h"

namespace rmpcpp {
/*
 * Struct that holds a discretized and integrated
 *  point in a RMP trajectory.
 */
template <class TSpace>
struct TrajectoryPointRMP {
  using Vector = Eigen::Matrix<double, TSpace::dim, 1>;
  Vector position;
  Vector velocity;
  Vector acceleration;
  double cumulative_length = 0.0;  // Cumulative length of this trajectory

  static std::string getHeaderFormat();
  std::string format() const;
};

/*
 * Class that holds a full trajectory (vector of points)
 */
template <class TSpace>
class TrajectoryRMP : public mjpc::Trajectory {
  using Vector = Eigen::Matrix<double, TSpace::dim, 1>;

 public:
  TrajectoryPointRMP<TSpace> start() const {
    return (trajectory_data_.size() > 0) ? trajectory_data_[0] : TrajectoryPointRMP<TSpace>();
  }
  TrajectoryPointRMP<TSpace> current() const {
    return (trajectory_data_.size() > 0) ? trajectory_data_.back() : TrajectoryPointRMP<TSpace>();
  }

  void addPoint(const Vector& p, const Vector& v, const Vector& a = Vector::Zero());

  int getSegmentCount() const;
  double getSmoothness() const;
  double getLength() const;

  inline const TrajectoryPointRMP<TSpace> operator[](int i) const {
    return trajectory_data_[i];
  };
  inline TrajectoryPointRMP<TSpace>& operator[](int i) {
    return trajectory_data_[i];
  };

  void clearData() {
    trajectory_data_.clear();
  }

  void writeToStream(std::ofstream& file) const;

 private:
  std::vector<TrajectoryPointRMP<TSpace>> trajectory_data_;
};

}  // namespace rmpcpp

#endif  // RMPCPP_PLANNER_TRAJECTORY_RMP_H
