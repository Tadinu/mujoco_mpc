#pragma once
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>
#include <stdexcept>

#include <Eigen/Core>

// absl
#include <absl/status/status.h>
#include <absl/status/statusor.h>
#include <absl/strings/str_join.h>
#include <absl/types/span.h>

// dm_robotics
#include <dm_robotics/least_squares_qp/core/lsqp_constraint.h>
#include <dm_robotics/least_squares_qp/core/lsqp_stack_of_tasks_solver.h>
#include <dm_robotics/least_squares_qp/core/lsqp_task.h>
#include <dm_robotics/least_squares_qp/core/lsqp_task_hierarchy.h>
#include <dm_robotics/least_squares_qp/core/utils.h>

// mjpc
#include "mjpc/planners/lsqp/lsqp_config.h"

namespace mjpc {
constexpr int kNumDof = 3;
constexpr double kDesiredValue = 1.23;
constexpr double kUpperBound = 1.0;
constexpr double kLowerBound = -std::numeric_limits<double>::infinity();

// Task to set all DoFs to kDesiredValue.
class LsqpCoreTask : public dm_robotics::LsqpTask {
public:
  LsqpCoreTask()
    : bias_(kNumDof, kDesiredValue), coefficient_matrix_(kNumDof * kNumDof) {
    Eigen::Map<Eigen::MatrixXd>(coefficient_matrix_.data(), kNumDof, kNumDof)
        .setIdentity();
  }

  absl::Span<const double> GetCoefficientMatrix() const override {
    return coefficient_matrix_;
  }

  absl::Span<const double> GetBias() const override { return bias_; }

  int GetNumberOfDof() const override { return kNumDof; }

  int GetBiasLength() const override { return kNumDof; }

private:
  std::vector<double> bias_;
  std::vector<double> coefficient_matrix_;
};

// Constraint for all variables between kLowerBound and kUpperBound.
class LsqpCoreConstraint final : public dm_robotics::LsqpConstraint {
public:
  LsqpCoreConstraint()
    : lower_bound_(kNumDof, kLowerBound),
      upper_bound_(kNumDof, kUpperBound),
      coefficient_matrix_(kNumDof * kNumDof) {
    Eigen::Map<Eigen::MatrixXd>(coefficient_matrix_.data(), kNumDof, kNumDof)
        .setIdentity();
  }

  LsqpCoreConstraint(const LsqpCoreConstraint&) = delete;
  LsqpCoreConstraint& operator=(const LsqpCoreConstraint&) = delete;

  absl::Span<const double> GetCoefficientMatrix() const override {
    return coefficient_matrix_;
  }

  absl::Span<const double> GetUpperBound() const override {
    return upper_bound_;
  }

  absl::Span<const double> GetLowerBound() const override {
    return lower_bound_;
  }

  int GetNumberOfDof() const override { return kNumDof; }

  int GetBoundsLength() const override { return kNumDof; }

private:
  const std::vector<double> lower_bound_;
  const std::vector<double> upper_bound_;
  std::vector<double> coefficient_matrix_;
};

static void lsqp_demo() {
  // Instantiate solver
  dm_robotics::LsqpStackOfTasksSolver qp_solver(
      dm_robotics::LsqpStackOfTasksSolver::Parameters{
          /*use_adaptive_rho=*/false,
                               /*return_error_on_nullspace_failure=*/true,
                               /*verbosity=*/
                               dm_robotics::LsqpStackOfTasksSolver::Parameters::VerboseFlags::kNone,
                               /*absolute_tolerance=*/1.0e-6,
                               /*relative_tolerance=*/0.0,
                               /*hierarchical_projection_slack=*/1.0e-6,
                               /*primal_infeasibility_tolerance=*/1.0e-8,
                               /*dual_infeasibility_tolerance=*/1.0e-8});

  // Create a task hierarchy to hold the task.
  dm_robotics::LsqpTaskHierarchy* task_hierarchy =
      qp_solver.AddNewTaskHierarchy(/*max_iterations*/ 10000);

  // Add task to solver.
  // [MyTask*, bool]
  auto [task_ptr, is_task_inserted] = task_hierarchy->InsertOrAssignTask(
      /*name*/ "MyTaskName",
               /*task*/ absl::make_unique<LsqpCoreTask>(),
               /*weight*/ 1.0,
               /*should_ignore_nullspace*/ false);
  CHECK(task_ptr != nullptr);
  CHECK(is_task_inserted);

  // Add constraint to solver.
  // [MyConstraint*, bool]
  auto [constraint_ptr, is_constraint_inserted] =
      qp_solver.InsertOrAssignConstraint(/*name*/ "LsqpCoreConstraintName",
                                                  /*constraint*/ absl::make_unique<LsqpCoreConstraint>());
  CHECK(constraint_ptr != nullptr);
  CHECK(is_constraint_inserted);

  // Setup and solve.
  absl::StatusOr<absl::Span<const double>> solution_or =
      qp_solver.SetupAndSolve();
  CHECK_EQ(solution_or.status(), absl::OkStatus())
      << "Failed to solve problem.";
  std::vector<double> solution = dm_robotics::AsCopy(*solution_or);
  std::cout << "Found solution: "
      << absl::StrJoin(solution.begin(), solution.end(), ", ")
      << std::endl;
}
} // end namespace mjpc