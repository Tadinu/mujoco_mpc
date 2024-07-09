#pragma once

#include "mjpc/planners/fabrics/include/fab_common.h"
#include "mjpc/planners/fabrics/include/fab_geometry_primitives.h"

using FabSelfCollisionPairs = std::map<std::string/*link_name*/,
                                       std::array<std::string, 2>/*paired_links_names*/>;
using FabCollisionLinks = std::map<std::string /*link_name*/, FabGeometricPrimitivePtr /*link_geom*/>;

struct FabCollisionLinkNotExistError : public std::runtime_error {
  explicit FabCollisionLinkNotExistError(const std::string& link_name)
    : std::runtime_error(std::string("Collision link not existing: ") + link_name) {
  }

  explicit FabCollisionLinkNotExistError(const char* link_name)
    : std::runtime_error(std::string("Collision link not existing: ") + link_name) {
  }
};

struct FabCollisionLinkUndefinedError : public std::runtime_error {
  explicit FabCollisionLinkUndefinedError(const std::string& link_name)
    : std::runtime_error(std::string("Collision link undefined: ") + link_name) {
  }

  explicit FabCollisionLinkUndefinedError(const char* link_name)
    : std::runtime_error(std::string("Collision link undefined: ") + link_name) {
  }
};

class FabRobotRepresentation {
public:
  FabRobotRepresentation() = default;

  FabRobotRepresentation(FabCollisionLinks collision_links, FabSelfCollisionPairs self_collision_pairs)
    : collision_links_(std::move(collision_links)), self_collision_pairs_(std::move(self_collision_pairs)) {
    check_self_collision_pairs();
  }

  void check_self_collision_pairs() const {
    for (const auto& [link_name, paired_links_names] : self_collision_pairs_) {
      if (!collision_links_.contains(link_name)) {
        throw FabCollisionLinkUndefinedError(link_name);
      }
      for (const auto& paired_link_name : paired_links_names) {
        if (!collision_links_.contains(paired_link_name)) {
          throw FabCollisionLinkUndefinedError(paired_link_name);
        }
      }
    }
  }

  FabCollisionLinks collision_links() const { return collision_links_; }

  FabGeometricPrimitivePtr collision_link(const std::string& link_name) const {
    if (collision_links_.contains(link_name)) {
      return collision_links_.at(link_name);
    }
    throw FabCollisionLinkNotExistError(link_name);
  }

  FabSelfCollisionPairs self_collision_pairs() const { return self_collision_pairs_; }

  std::array<std::string, 2> self_collision_pair(const std::string& link_name) {
    if (self_collision_pairs_.contains(link_name)) {
      return self_collision_pairs_.at(link_name);
    }
    throw FabCollisionLinkNotExistError(link_name);
  }

protected:
  FabCollisionLinks collision_links_;
  FabSelfCollisionPairs self_collision_pairs_;
};