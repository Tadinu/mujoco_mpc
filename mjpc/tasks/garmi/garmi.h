#pragma once

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/tasks/lsqp/lsqp.h"

namespace mjpc {
class Garmi : public Lsqp {
  using LsqpSolverPtr = Lsqp::LsqpSolverPtr;

public:
  std::string Name() const override { return "Garmi"; }

  std::string XmlPath() const override {
    return GetModelPath("garmi/garmi_scene.xml");
  }

  bool IsBimanualSupported() const override { return true; }

  std::string GetBaseBodyName() const override {
    return "torso";
  }

  Garmi() : residual_(this) {
  }

  void TransitionLocked(mjModel* model, mjData* data) override;

  int GetTargetObjectId() const override { return QueryBodyId("target"); }
  const mjtNum* GetGoalPos() const override { return QueryBodyMocapPos("target"); }
  const mjtNum* GetGoalVel() const override { return QueryTargetVel(); }
  const mjtNum* GetGoalAcc() const override { return QueryTargetAcc(); }

  std::vector<FabSubGoalPtr> GetSubGoals() const override {
    // Static subgoals with static [desired_state.pos]
    static std::vector<FabSubGoalPtr> subgoals = {std::make_shared<FabStaticSubGoal>(
                                                      FabSubGoalConfig{.name = "subgoal0",
                                                                       .type = FabSubGoalType::STATIC,
                                                                       .is_primary_goal = true,
                                                                       .epsilon = 0.02,
                                                                       .indices = {0, 1, 2},
                                                                       .weight = 0.4,
                                                                       .parent_link_name = "left_link0",
                                                                       .child_link_name =
                                                                       "left_leftfinger"}),
                                                  std::make_shared<FabStaticSubGoal>(
                                                      FabSubGoalConfig{.name = "subgoal1",
                                                                       .type = FabSubGoalType::STATIC,
                                                                       .is_primary_goal = true,
                                                                       .epsilon = 0.02,
                                                                       .indices = {0, 1, 2},
                                                                       .weight = 0.4,
                                                                       .parent_link_name = "right_link0",
                                                                       .child_link_name =
                                                                       "right_leftfinger"})
    };
    auto& subgoal0_cfg = subgoals[0]->cfg_;
    if (!IsGoalFixed()) {
      subgoal0_cfg.type = FabSubGoalType::DYNAMIC;
    }
    subgoal0_cfg.desired_state = GetGoalState();
    if (subgoal0_cfg.desired_state.pose.empty()) {
      subgoal0_cfg.desired_state.reset();
    }
    // Take goal's rot as the desired_state's rot offset
    subgoal0_cfg.desired_state.pose_offset =
        FabPose{.pos = {0., 0., 0.}, .rot = subgoal0_cfg.desired_state.pose.rot};
    return subgoals;
  }

  mjModel* ConstructModel() override {
    char error[1024];
    const auto xml_path = XmlPath(); //MUJOCO_DIR + "/mujoco_mpc/mjpc/tasks/garmi/garmi_scene.xml";
    mjpc::print("Loading model from:", xml_path);
    mjModel* model = mj_loadXML(xml_path.c_str(), vfs_.get(), error, 1024);
    return model;
  }

  void InitMocaps() override {
    MoveBodyMocapToSite("left_target", "left_ee_site");
    MoveBodyMocapToSite("right_target", "right_ee_site");
  }

  void InitSolverConfigs(const LsqpSolverPtr& solver, const mjData* data, int ndofs) override;
  std::vector<double> Control(double* policy_action, mjData* data,
                              const LsqpSolverPtr& solver) override;
  std::vector<double> Solve(const LsqpSolverPtr& solver, const mjData* data) override;

protected:
  class ResidualFn : public BaseResidualFn {
  public:
    explicit ResidualFn(const Garmi* task) : BaseResidualFn(task) {
    }

    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  std::unique_ptr<mjpc::AbstractResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }

  ResidualFn* InternalResidual() override { return &residual_; }
  double last_solve_time_ = 0;

private:
  ResidualFn residual_;
};
} // namespace mjpc
