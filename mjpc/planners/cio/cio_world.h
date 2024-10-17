#pragma once

class CIORectangle : public CIOObject {
  CIORectangle(double width, double height, const Eigen::Vector3d& pos,
               const Eigen::Vector3d& linear_vel = Eigen::Vector3d::Zero(), double step_size = 0.5)
      : CIOObject(CIOPose(pos), linear_vel, step_size) : width_(width), height_(height) {
    // pose = Pose(pos.x, pos.y, 0.0) # not using orientations yet
    // vel = Velocity(vel.x, vel.y, 0.0
    lines_ = make_lines();  // rectangles are made up of 4 line objects
  }

  // defines list of lines in clockwise starting from left line
  void make_lines() {
    std::vector<Eigen::Vector3d> lines;
    pose_ = self.pose;
    rpy = self.pose.rpy;
    length_ = self.height;
    for (auto i = 0; i < 4; ++i) {
      line = Line(pose_, rpy, length_);
      lines += [line];
      (_, pose) = line.get_endpoints();
      angle = angle - np.pi / 2;
      if (length_ == height_) {
        length = width_;
      } else {
        length = height_;
      }
    }
    return lines;
  }

  // return the closest projected point out of all rect surfaces
  // use a softmin instead of a hard min to make function smooth
  void project_point(const Eigen::Vector3d& point) {
    k = 1.e4;
    num_sides = len(self.lines);
    p_nearest = np.zeros((num_sides, 2));
    for (auto j = 0; j < num_sides; ++j) {
      p_nearest [j, :] = self.lines[j].project_point(point);
    }
    p_mat = np.tile(point.T, (num_sides, 1));
    ones_vec = np.ones(num_sides);
    nu = np.divide(ones_vec, ones_vec + np.linalg.norm(p_mat - p_nearest, axis = 1) * *2 * k);
    nu = nu / sum(nu);
    nu = np.tile(nu, (2, 1)).T;
    closest_point = sum(np.multiply(nu, p_nearest));
    return closest_point;
  }
};

class CIOSphere : public CIOObject {
  CIOSphere(double radius, const CIOPosition& pos, const CIOLinearVelocity& vel, double step_size = 0.5)
      : CIOObject(pose, vel, step_size), radius_(radius) {
    pose = CIOPose(pos.x, pos.y, 0.0) vel = CIOVelocity(vel.x, vel.y, 0.0)
  }

  // Projects the given point onto the surface of this object
  void project_point(const Eigen::Vector3d& point) {
    origin_to_point = np.subtract(point[:2], np.array([ self.pose.x, self.pose.y ])) origin_to_point /=
        np.linalg.norm(origin_to_point) closest_point =
            np.array([ self.pose.x, self.pose.y ]) + (origin_to_point * self.radius) return closest_point
  }
};
