#pragma once

#include <casadi/casadi.hpp>
#include <cassert>
#include <memory>

#include "mjpc/planners/fabrics/include/fab_casadi_function.h"
#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_core_util.h"
#include "mjpc/planners/fabrics/include/fab_diff_map.h"
#include "mjpc/planners/fabrics/include/fab_energy.h"
#include "mjpc/planners/fabrics/include/fab_geometry.h"
#include "mjpc/planners/fabrics/include/fab_math_util.h"
#include "mjpc/planners/fabrics/include/fab_spectral_semi_sprays.h"
#include "mjpc/planners/fabrics/include/fab_variables.h"

class FabSphere;
class FabCuboid;
class FabPlane;

class FabGeometricPrimitive {
 public:
#pragma region FabDistanceUndefinedError
  class FabDistanceUndefinedError : public std::runtime_error {
   public:
    explicit FabDistanceUndefinedError(const std::string& error_msg) : std::runtime_error(error_msg) {}

    explicit FabDistanceUndefinedError(const char* error_msg) : std::runtime_error(error_msg) {}

    static FabDistanceUndefinedError customized(const FabGeometricPrimitive* prim1,
                                                const FabGeometricPrimitive* prim2) {
      FabDistanceUndefinedError error("");
      error.prim1_ = prim1;
      error.prim2_ = prim2;
      return error;
    }

    const char* what() const _NOEXCEPT override {
      static std::string full_message;
      if (prim1_ && prim2_) {
        full_message = std::string("Distance undefined between ") + prim1_->name_ + (" & ") + prim2_->name_;
      } else {
        assert(false);
      }
      return full_message.c_str();
    }

    const FabGeometricPrimitive* prim1_ = nullptr;
    const FabGeometricPrimitive* prim2_ = nullptr;
  };
#pragma endregion FabDistanceUndefinedError

 public:
  FabGeometricPrimitive() = default;

  explicit FabGeometricPrimitive(std::string name) : name_(std::move(name)) {}

  std::string to_string() const { return std::string(typeid(this).name()) + ": " + name_; }

  CaSX position() const {
    CaSX position;
    origin_.get(position, false, CaSlice(0, 3), CaSlice(3));
    return position;
  }

  void set_position(const CaSX& position, bool free = false) {
    origin_.set(position, false, CaSlice(0, 3), CaSlice(3));
    if (free) {
      CaSX position_0;
      position.get(position_0, false, CaSlice(0, 0));
      const std::string position_0_name = position_0.name();
      sym_parameters_[position_0_name.substr(position_0_name.size() - 2, 2)] = position;
    }
  }

  CaSX origin() const { return origin_; }

  void set_origin(CaSX origin, const bool free = false) {
    origin_ = std::move(origin);
    if (free) {
      CaSX origin_0;
      origin_.get(origin_0, false, CaSlice(0, 0));
      const std::string origin_0_name = origin_0.name();
      sym_parameters_[origin_0_name.substr(origin_0_name.size() - 2, 2)] = origin_;
    }
  }

  std::string name() const { return name_; }

  virtual std::vector<double> size() const { return {}; }

  CaSXDict sym_parameters() {
    for (const auto& [sym_size_key, sym_size_value] : sym_size()) {
      sym_parameters_.insert_or_assign(sym_size_key, sym_size_value);
    }
    return sym_parameters_;
  }

  FabNamedMap<double, std::vector<double>> parameters() const { return parameters_; }

  virtual CaSXDict sym_size() const { return {}; }

  virtual CaSX distance(const FabGeometricPrimitive* primitive) = 0;

 protected:
  std::string name_;
  CaSX position_;
  CaSX origin_ = fab_math::CASX_TRANSF_IDENTITY;
  FabNamedMap<double, std::vector<double>> parameters_;
  CaSXDict sym_parameters_;
};

class FabCapsule : public FabGeometricPrimitive {
 public:
  FabCapsule() = default;

  FabCapsule(std::string name, double radius, double length)
      : FabGeometricPrimitive(std::move(name)), radius_(radius), length_(length) {
    sym_radius_ = CaSX::sym(std::string("radius_") + name_, 1);
    sym_length_ = CaSX::sym(std::string("length_") + name_, 1);
    parameters_[std::string("radius_") + name_] = radius;
    parameters_[std::string("length_") + name_] = length;
  }

  virtual ~FabCapsule() = default;

  std::vector<double> size() const override { return {radius_, length_}; }

  CaSXDict sym_size() const override {
    return {{sym_radius_.name(), sym_radius_}, {sym_length_.name(), sym_length_}};
  }

  double radius() const { return radius_; }

  double length() const { return length_; }

  CaSX sym_radius() const { return sym_radius_; }

  CaSX sym_length() const { return sym_length_; }

  CaSXVector center() const {
    auto tf_origin_center_0 = fab_math::CASX_TRANSF_IDENTITY;
    tf_origin_center_0.set(0.5 * sym_length_, false, CaSlice(2, 3));
    const auto tf_center_0 = CaSX::mtimes(origin_, tf_origin_center_0);
    auto tf_origin_center_1 = fab_math::CASX_TRANSF_IDENTITY;
    tf_origin_center_1.set(-0.5 * sym_length_, false, CaSlice(2, 3));
    const auto tf_center_1 = CaSX::mtimes(origin_, tf_origin_center_1);
    CaSX center_0, center_1;
    tf_center_0.get(center_0, false, CaSlice(0, 3), CaSlice(3));
    tf_center_1.get(center_1, false, CaSlice(0, 3), CaSlice(3));
    return {center_0, center_1};
  }

  CaSX distance(const FabGeometricPrimitive* primitive) override;

 protected:
  double radius_ = 0.;
  double length_ = 0.;
  CaSX sym_radius_;
  CaSX sym_length_;
};

class FabSphere : public FabGeometricPrimitive {
 public:
  FabSphere() = default;

  FabSphere(std::string name, double radius) : FabGeometricPrimitive(std::move(name)), radius_(radius) {
    sym_radius_ = CaSX::sym(std::string("radius_") + name_, 1);
    parameters_[std::string("radius_") + name_] = radius;
  }

  virtual ~FabSphere() = default;

  std::vector<double> size() const override { return {radius_}; }

  CaSXDict sym_size() const override { return {{sym_radius_.name(), sym_radius_}}; }

  double radius() const { return radius_; }

  CaSX sym_radius() const { return sym_radius_; }

  CaSX distance(const FabGeometricPrimitive* primitive) override;

 protected:
  double radius_ = 0.;
  CaSX sym_radius_;
};

class FabCuboid : public FabGeometricPrimitive {
 public:
  FabCuboid() = default;

  FabCuboid(std::string name, std::vector<double> size)
      : FabGeometricPrimitive(std::move(name)), size_(std::move(size)) {
    sym_size_ = CaSX::sym(std::string("size_") + name_, 3);
    parameters_[std::string("size_") + name_] = size;
  }

  virtual ~FabCuboid() = default;

  std::vector<double> size() const override { return size_; }

  CaSXDict sym_size() const override {
    CaSX size_0;
    sym_size_.get(size_0, false, CaSlice(0, 0));
    const std::string size_0_name = size_0.name();
    return {{size_0_name.substr(size_0_name.size() - 2, 2), sym_size_}};
  }

  CaSX distance(const FabGeometricPrimitive* primitive) override {
    throw FabDistanceUndefinedError::customized(this, primitive);
  }

 protected:
  std::vector<double> size_ = {0., 0., 0.};
  CaSX sym_size_;
};

class FabPlane : public FabGeometricPrimitive {
 public:
  FabPlane() = default;

  FabPlane(std::string name, std::vector<double> plane_equation = {0, 0, 0, 1})
      : FabGeometricPrimitive(std::move(name)), plane_equation_(std::move(plane_equation)) {
    sym_plane_equation_ = CaSX::sym(name_, 4);
    parameters_[name_] = plane_equation;
  }

  virtual ~FabPlane() = default;

  std::vector<double> size() const override { return plane_equation_; }

  CaSXDict sym_size() const override { return {{name_, sym_plane_equation_}}; }

  CaSX sym_plane_equation() const { return sym_plane_equation_; }

  CaSX distance(const FabGeometricPrimitive* primitive) override {
    throw FabDistanceUndefinedError::customized(this, primitive);
  }

 protected:
  std::vector<double> plane_equation_ = {0., 0., 0., 1.};
  CaSX sym_plane_equation_;
};

using FabGeometricPrimitivePtr = std::shared_ptr<FabGeometricPrimitive>;
