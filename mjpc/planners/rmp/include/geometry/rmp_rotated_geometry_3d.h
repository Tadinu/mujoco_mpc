//
// Created by mpantic on 06.09.22.
//

#ifndef RMPCPP_ROTATEDGEOMETRY3D_H
#define RMPCPP_ROTATEDGEOMETRY3D_H
#include "mjpc/planners/rmp/include/core/rmp_base_geometry.h"
namespace rmpcpp {
class RotatedGeometry3d : public RMPBaseGeometry<3, 3> {
 public:
  // type alias for readability.
  using base = RMPBaseGeometry<3, 3>;
  using VectorX = base::VectorX;
  using Vector = base::VectorQ;
  using StateX = base::StateX;
  using StateQ = base::StateQ;
  using J_phi = base::J_phi;

  inline void setRotation(const Eigen::Matrix3d &rotation) {
    R_x_q_ = rotation;
  }

  /**
   * Return jacobian. simply the rotation matrix;
   */
  inline virtual J_phi J(const StateX&) const { return R_x_q_; }

  inline virtual VectorX convertPosToX(const VectorQ &vector_q) const { return R_x_q_ * vector_q; }
  inline virtual StateX convertToX(const StateQ &state_q) const {
    return {.pos_ = R_x_q_ * state_q.pos_,
            .vel_ = R_x_q_ * state_q.vel_};
  }

  inline virtual VectorQ convertPosToQ(const VectorX &vector_x) const { return R_x_q_.transpose() * vector_x; }
  inline virtual StateQ convertToQ(const StateX &state_x) const {
    return {.pos_ = R_x_q_.transpose() * state_x.pos_,
            .vel_ = R_x_q_.transpose() * state_x.vel_};
  }

 private:
  Eigen::Matrix3d R_x_q_{Eigen::Matrix3d::Identity()};
};

}  // namespace rmpcpp

#endif  // RMPCPP_ROTATEDGEOMETRY3D_H
