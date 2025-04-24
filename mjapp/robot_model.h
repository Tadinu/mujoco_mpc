#pragma once

#include <absl/strings/match.h>
#include <mujoco/mujoco.h>

#include <filesystem>

// mjpc
#include "mjpc/mjcf/mjcf_model.h"
#include "mjpc/utilities.h"

namespace mjapp {
class RobotModel {
public:
  RobotModel() = default;

  RobotModel(std::string name, std::string model_path, std::string base_link_name,
             std::vector<std::string> endtip_names = {})
    : name_(std::move(name)), model_path_(std::move(model_path)),
      base_link_name_(std::move(base_link_name)),
      endtip_names_(std::move(endtip_names)),
      mjcf_model_(std::make_shared<mjpc::MjcfModel>()) {
    if (absl::EndsWithIgnoreCase(model_path_, ".xml") ||
        absl::EndsWithIgnoreCase(model_path_, ".mjcf") ||
        absl::StartsWith(model_path_, "<mujoco")) {
      // Assume [model_path] is path to [.xml/.mjcf] or xml string itself
      ReadEntityModel();
    } else {
      mjpc::print("[mjapp::RobotModel] Unsupported model type", model_path_);
    }
  }

  std::string Name() const { return name_; }

  std::string ModelName() const {
    return IsLoadedFromXMLString() ? name_ : std::filesystem::path(model_path_).stem().string();
  }

  mjpc::MjcfModelPtr MJCFModel() const { return mjcf_model_; }

  bool IsLoadedFromXMLString() const {
    return !model_path_.empty() && !absl::EndsWithIgnoreCase(model_path_, ".xml") &&
           !absl::EndsWithIgnoreCase(model_path_, ".mjcf");
  }

  bool ReadEntityModel() const {
    if (false == ReadMJCF()) {
      mjpc::print("[mjapp::RobotModel] failed reading MODEL", model_path_);
      return false;
    }
    return true;
  }

  bool ReadMJCF() const {
    mjcf_model_->base_link_name = base_link_name_;
    mjcf_model_->endtip_names = endtip_names_;

    return IsLoadedFromXMLString()
             ? mjcf_model_->FromMjcfStr(model_path_)
             : mjcf_model_->FromMjcfFile(model_path_);
  }

  std::vector<std::string> JointNames() const {
    return mjcf_model_->actuated_joint_names;
  }

  std::vector<std::string> LinkNames() const {
    return mjcf_model_->LinkNames();
  }

  void Step(bool kinematics_only = MJPC_LSQP_KINEMATICS_ONLY) {
    auto* m = mjcf_model_->model;
    auto* d = mjcf_model_->data;
    if (kinematics_only) {
      mj_kinematics(m, d);
      mj_comPos(m, d);
    } else {
      mj_step(m, d);
    }
    if (m->neq > 0) {
      mj_makeConstraint(m, d);
    }
  }

  void ApplyCtrl(int ctrl_id, const mjtNum ctrl) {
    if (ctrl_id < mjcf_model_->model->nu) {
      mjcf_model_->data->ctrl[ctrl_id] = ctrl;
    }
  }

  void ApplyCtrls(const mjtNum* ctrl, int num, int start = 0) {
    mju_copy(mjcf_model_->data->ctrl + start, ctrl, num);
  }

protected:
  mjpc::MjcfModelPtr mjcf_model_ = nullptr;
  std::string name_;
  std::string model_path_;
  std::string base_link_name_;
  std::vector<std::string> endtip_names_;
};

using RobotModelPtr = std::shared_ptr<RobotModel>;
} // end namespace mjapp