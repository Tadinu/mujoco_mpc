# pragma once

#include <Eigen/Dense>

#define MJPC_PLANNER_LSQP_DEBUG (0)

namespace mjpc {
static Eigen::Matrix3d Skew(const Eigen::Vector3d& v) {
  return Eigen::Matrix3d{
      {0.0, -v.z(), v.y()},
      {v.z(), 0.0, -v.x()},
      {-v.y(), v.x(), 0.0}};
}

// Abstract class for Matrix Lie Groups
template <typename TLieGroup>
class LsqpMatrixLieGroup {
public:
  virtual ~LsqpMatrixLieGroup() = default;

  virtual void PrintSelf(const std::string& label) const = 0;
  // Accessors for dimensions of the Lie group
  virtual int MatrixDim() const = 0;
  virtual int ParametersDim() const = 0;
  virtual int Dim() const = 0; // Dimension of the transformed coordinates
  virtual int TangentDim() const = 0; // Dimension of the tangent space

  // Factory methods to be implemented by derived classes
  virtual TLieGroup Identity() const = 0; // Returns the identity element
  virtual TLieGroup FromMatrix(const Eigen::MatrixXd& matrix) const = 0;
  // From matrix to group member
  virtual TLieGroup SampleUniform() const = 0; // Draw a uniform sample from this Lie group

  // Accessors for derived classes to define their specific implementations
  virtual Eigen::MatrixXd AsMatrix() const = 0; // Return the group element as a matrix
  virtual Eigen::VectorXd Parameters() const = 0; // Return the underlying Lie group parameters

  // Group operations - implementation depends on the specific subclass
  virtual Eigen::VectorXd Apply(const Eigen::VectorXd& target) const = 0; // Applies group action to a point
  virtual TLieGroup Multiply(const TLieGroup& other) const = 0; // Compose transformations
  TLieGroup operator*(const TLieGroup& other) const {
    return Multiply(other);
  }

  virtual TLieGroup Inverse() const = 0; // Compute the inverse
  virtual TLieGroup Normalize() const = 0; // Normalize the group member

  // Logarithm and Exponential maps for Lie Groups
  virtual Eigen::VectorXd Log() const = 0;
  virtual TLieGroup Exp(const Eigen::VectorXd& tangent) const = 0;

  // Adjoint transformation of the matrix Lie group
  virtual Eigen::MatrixXd Adjoint() const = 0;

  // Plus and Minus operations defined on Lie groups
  // Eq 25
  virtual TLieGroup RPlus(const Eigen::VectorXd& tangent) const {
    return *this * Exp(tangent);
  }

  // Eq 26
  virtual Eigen::VectorXd RMinus(const TLieGroup& other) const {
    const TLieGroup composed = other.Inverse() * static_cast<const TLieGroup&>(*this);
    return composed.Log();
  }

  // Eq 27
  virtual TLieGroup LPlus(const Eigen::VectorXd& tangent) const {
    return Exp(tangent) * static_cast<const TLieGroup&>(*this);
  }

  // Eq 28
  virtual Eigen::VectorXd LMinus(const TLieGroup& other) const {
    const TLieGroup composed = static_cast<const TLieGroup&>(*this) * other.Inverse();
    return composed.Log();
  }

  // Alias for "plus" and "minus" operations
  virtual TLieGroup Plus(const Eigen::VectorXd& tangent) const {
    return RPlus(tangent);
  }

  virtual Eigen::VectorXd Minus(const TLieGroup& other) const {
    return RMinus(other);
  }

  // Jacobian computations
  virtual Eigen::MatrixXd LeftJac(const Eigen::VectorXd& tangent) const = 0;
  virtual Eigen::MatrixXd LeftJacInverse(const Eigen::VectorXd& tangent) const = 0;

  // Right Jacobian and its inverse - Eq 67
  virtual Eigen::MatrixXd RightJac(const Eigen::VectorXd& tangent) const {
    return LeftJac(-tangent);
  }

  virtual Eigen::MatrixXd RightJacInverse(const Eigen::VectorXd& tangent) const {
    return LeftJacInverse(-tangent);
  }

  // Jacobian of the logarithm map - Eq 79
  virtual Eigen::MatrixXd JacLog() const {
    return RightJacInverse(Log());
  }
};
}
