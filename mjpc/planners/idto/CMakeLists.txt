# drake
find_package(drake CONFIG REQUIRED)
# /opt/drake/lib/cmake/drake/drake-config.cmake
message("DRAKE: ${drake_VERSION}")

set(IDTO_HEADERS
        idto_common.h
        idto_pd_plus_controller.h
        idto_mpc.h
        idto_planner.h
        optimizer/convergence_criteria_tolerances.h
        optimizer/inverse_dynamics_partials.h
        optimizer/penta_diagonal_matrix.h
        optimizer/penta_diagonal_solver.h
        optimizer/problem_definition.h
        optimizer/solver_parameters.h
        optimizer/trajectory_optimizer.h
        optimizer/trajectory_optimizer_solution.h
        optimizer/trajectory_optimizer_state.h
        optimizer/trajectory_optimizer_workspace.h
        optimizer/velocity_partials.h
        optimizer/warm_start.h
        utils/profiler.h
)

target_sources(
        libmjpc PRIVATE
        idto_pd_plus_controller.cc
        idto_mpc.cc
        idto_planner.cc
        optimizer/inverse_dynamics_partials.cc
        optimizer/penta_diagonal_matrix.cc
        optimizer/trajectory_optimizer.cc
        optimizer/trajectory_optimizer_solution.cc
        optimizer/trajectory_optimizer_state.cc
        optimizer/trajectory_optimizer_workspace.cc
        optimizer/velocity_partials.cc
        utils/profiler.cc
        ${IDTO_HEADERS}
)
target_link_libraries(
        libmjpc
        drake::drake
        drake::drake-lcmtypes-cpp
        drake::drake-marker
)
target_include_directories(libmjpc
        PUBLIC
        ${drake_INCLUDE_DIR}
)
