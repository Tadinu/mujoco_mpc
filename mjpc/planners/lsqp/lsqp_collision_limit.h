#pragma once

#include <stdexcept>
#include <iostream>
#include <utility>
#include <vector>

// Eigen
#include <Eigen/Dense>

// MuJoCo
#include <mujoco/mujoco.h>

// mjpc
#include "mjpc/planners/lsqp/lsqp_limit.h"
#include "mjpc/utils/mjpc_ctrl_util.h"

namespace mjpc {
// Ref: https://github.com/kevinzakka/mink/blob/main/mink/limits/collision_avoidance_limit.py

using LsqpGeomId = std::variant<int, std::string>;
using LsqpGeomIdList = std::vector<LsqpGeomId>;
using LsqpCollisionPair = std::pair<LsqpGeomIdList, LsqpGeomIdList>;
using LsqpCollisionPairList = std::vector<LsqpCollisionPair>;

struct LsqpContact {
  double dist = 0;
  Vector6d fromto;
  int geom1 = -1;
  int geom2 = -1;
  double distmax = 0;

  Eigen::Vector3d normal() const {
    Eigen::Vector3d norm = fromto.tail<3>() - fromto.head<3>();
    mju_normalize3(norm.data());
    return norm;
  }

  bool active() const {
    return dist <= distmax;
  }
};

class LsqpCollisionLimit : public LsqpLimit {
public:
  LsqpCollisionLimit() = default;

  LsqpCollisionLimit(const mjModel* model, int ndofs, const LsqpCollisionPairList& geom_pairs,
                     double gain = 0.85,
                     double minimum_distance_from_collisions = 0.005,
                     double collision_detection_distance = 0.01,
                     double bound_relaxation = 0.0)
    : LsqpLimit(model, ndofs), gain_(gain),
      minimum_distance_from_collisions_(minimum_distance_from_collisions),
      collision_detection_distance_(collision_detection_distance),
      bound_relaxation_(bound_relaxation) {
    geom_id_pairs_ = ConstructGeomIdPairs(geom_pairs);
    max_num_contacts_ = geom_id_pairs_.size();
  }

  LsqpConstraint ComputeQPInequalities(mjData* data, const LsqpConfig& config, double dt = 1.0) const {
    Eigen::VectorXd upper_bound = Eigen::VectorXd::Constant(max_num_contacts_,
                                                            std::numeric_limits<double>::infinity());
    Eigen::MatrixXd coefficient_matrix = Eigen::MatrixXd::Zero(max_num_contacts_, ndofs_);
    int idx = 0;
    for (const auto& [geom1_id, geom2_id] : geom_id_pairs_) {
      const auto contact = ComputeContactWithMinimumDistance(data, geom1_id, geom2_id);
      if (!contact.active()) {
        idx++;
        continue;
      }
      const auto hi_bound_dist = contact.dist;
      if (hi_bound_dist > minimum_distance_from_collisions_) {
        const auto dist = hi_bound_dist - minimum_distance_from_collisions_;
        upper_bound[idx] = (gain_ * dist / dt) + bound_relaxation_;
      } else {
        upper_bound[idx] = bound_relaxation_;
      }
      const auto jac = ComputeContactWithNormalJacobian(data, contact);
      coefficient_matrix.row(idx) = -jac.head(ndofs_);
      idx++;
    }
    return LsqpConstraint{.G = std::move(coefficient_matrix), .h = std::move(upper_bound)};
  }

protected:
  LsqpContact ComputeContactWithMinimumDistance(const mjData* data, int geom1_id, int geom2_id) const {
    // Returns the smallest signed distance between a geom pair
    Vector6d fromto;
    const auto dist = mj_geomDistance(model_, data,
                                      geom1_id, geom2_id, collision_detection_distance_, fromto.data());
    return LsqpContact(dist, fromto, geom1_id, geom2_id, collision_detection_distance_);
  }

  // Ref: https://github.com/google-deepmind/dm_robotics/blob/main/cpp/mujoco/src/utils.cc
  // The normal always points geom1 -> geom2
  // -> return: normal.transpose() * (jac2 - jac1)
  Eigen::VectorXd ComputeContactWithNormalJacobian(const mjData* data, const LsqpContact& contact) const {
    const auto geom1_body = model_->geom_bodyid[contact.geom1];
    const auto geom2_body = model_->geom_bodyid[contact.geom2];
    const Eigen::Vector3d geom1_contact_pos = contact.fromto.head<3>();
    const Eigen::Vector3d geom2_contact_pos = contact.fromto.tail<3>();

    Eigen::VectorXd jacobian(model_->nv);
    std::vector<double> jac_buffer(3 * model_->nv, 0);
    Eigen::Map<Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::RowMajor>>
        jacobian_buffer_map(jac_buffer.data(), 3, model_->nv);

    // Compute the Jacobian for the point in geom2, and project it into the
    // normal. Eigen's noalias is necessary to prevent dynamic memory allocation.
    mj_jac(model_, data, jacobian_buffer_map.data(), nullptr,
           geom2_contact_pos.data(), geom2_body);
    jacobian.noalias() = contact.normal().transpose() * jacobian_buffer_map;

    // Compute the Jacobian for the point in geom1, project it into the normal,
    // and subtract from the Jacobian for the point in geom2. This is the
    // resulting normal contact Jacobian.
    mj_jac(model_, data, jacobian_buffer_map.data(), nullptr,
           geom1_contact_pos.data(), geom1_body);
    jacobian.noalias() -= contact.normal().transpose() * jacobian_buffer_map;
    return jacobian;
  }

  // Query if the geoms are part of the same body, or if their bodies are welded together
  bool AreGeomsWeldedTogether(int geom_id1, int geom_id2) const {
    const auto body1 = model_->geom_bodyid[geom_id1];
    const auto body2 = model_->geom_bodyid[geom_id2];
    const auto weld1 = model_->body_weldid[body1];
    const auto weld2 = model_->body_weldid[body2];
    return weld1 == weld2;
  }

  // Query if the geom bodies have a parent-child relationship
  bool AreGeomBodiesParentChild(int geom_id1, int geom_id2) const {
    const auto body_id1 = model_->geom_bodyid[geom_id1];
    const auto body_id2 = model_->geom_bodyid[geom_id2];

    // body_weldid is the ID of the body's weld
    const auto body_weldid1 = model_->body_weldid[body_id1];
    const auto body_weldid2 = model_->body_weldid[body_id2];

    // weld_parent_id is the ID of the parent of the body's weld
    const auto weld_parent_id1 = model_->body_parentid[body_weldid1];
    const auto weld_parent_id2 = model_->body_parentid[body_weldid2];

    // weld_parent_weldid is the weld ID of the parent of the body's weld
    const auto weld_parent_weldid1 = model_->body_weldid[weld_parent_id1];
    const auto weld_parent_weldid2 = model_->body_weldid[weld_parent_id2];

    const auto cond1 = body_weldid1 == weld_parent_weldid2;
    const auto cond2 = body_weldid2 == weld_parent_weldid1;
    return cond1 || cond2;
  }

  // Query if the geoms pass the contype/conaffinity check
  bool IsContypeConaffinityCheckPassed(int geom_id1, int geom_id2) const {
    const bool cond1 = model_->geom_contype[geom_id1] & model_->geom_conaffinity[geom_id2];
    const bool cond2 = model_->geom_contype[geom_id2] & model_->geom_conaffinity[geom_id1];
    return cond1 || cond2;
  }

  LsqpGeomIdList HomogenizeGeomIdList(const LsqpGeomIdList& geom_list) const {
    // Take a heterogeneous list of geoms (specified via ID or name) and return a homogenous list of IDs (int)
    LsqpGeomIdList res;
    for (const auto& g : geom_list) {
      if (std::get_if<int>(&g)) {
        res.push_back(g);
      } else {
        res.emplace_back(mj_name2id(model_, mjOBJ_GEOM, std::get<std::string>(g).c_str()));
      }
    }
    return res;
  }

  std::vector<LsqpCollisionPair> CollisionPairtsToGeomIdPairs(
      const LsqpCollisionPairList& collision_pairs) const {
    LsqpCollisionPairList geom_id_pairs;
    for (const auto& collision_pair : collision_pairs) {
      const auto ids_A = HomogenizeGeomIdList(collision_pair.first);
      const auto ids_B = HomogenizeGeomIdList(collision_pair.second);
      geom_id_pairs.emplace_back(ids_A, ids_B);
    }
    return geom_id_pairs;
  }

  /* Returns a set of geom ID pairs for all possible geom-geom collisions.
   *
   * The contacts are added based on the following heuristics:
   *     1) Geoms that are part of the same body or weld are not included.
   *     2) Geoms where the body of one geom is a parent of the body of the other
   *         geom are not included.
   *     3) Geoms that fail the contype-conaffinity check are ignored.
   *
   * Note:
   *     1) If two bodies are kinematically welded together (no joints between them)
   *         they are considered to be the same body within this function.
  */
  std::map<int, int> ConstructGeomIdPairs(const LsqpCollisionPairList& geom_pairs) {
    std::map<int, int> res;
    const LsqpCollisionPairList geom_id_pairs = CollisionPairtsToGeomIdPairs(geom_pairs);
    for (const LsqpCollisionPair& geom_id_pair : geom_id_pairs) {
      for (const auto& geom_id_1 : geom_id_pair.first) {
        for (const auto& geom_id_2 : geom_id_pair.second) {
          const int geom_id_a = std::get<0>(geom_id_1);
          const int geom_id_b = std::get<0>(geom_id_2);
          const bool weld_body_cond = !AreGeomsWeldedTogether(geom_id_a, geom_id_b);
          const bool parent_child_cond = !AreGeomBodiesParentChild(geom_id_a, geom_id_b);
          const bool contype_conaffinity_cond = IsContypeConaffinityCheckPassed(geom_id_a, geom_id_b);
          if (weld_body_cond and parent_child_cond and contype_conaffinity_cond) {
            res.emplace(std::min(geom_id_a, geom_id_b), std::max(geom_id_a, geom_id_b));
          }
        }
      }
    }
    return res;
  }

private:
  double gain_ = 1.0;
  double minimum_distance_from_collisions_ = 0.005;
  double collision_detection_distance_ = 0.01;
  double bound_relaxation_ = 0.;
  int max_num_contacts_ = 0;
  std::map<int, int> geom_id_pairs_;
};
} // end namespace mjpc
