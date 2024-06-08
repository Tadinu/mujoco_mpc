# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(MJPC_INSTALL_PUBLIC 0)

find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

add_library(threadpool STATIC)
target_sources(
  threadpool
  PUBLIC threadpool.h
  PRIVATE threadpool.cc
)
target_link_libraries(
  threadpool
  absl::base
)
target_include_directories(threadpool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(LIB_MJPC_HEADERS
    states/state.h
    agent.h
    trajectory.h
    utilities.h
    tasks/tasks.h
    tasks/acrobot/acrobot.h
    tasks/allegro/allegro.h
    tasks/bimanual/handover/handover.h
    tasks/bimanual/reorient/reorient.h
    tasks/cartpole/cartpole.h
    tasks/fingers/fingers.h
    tasks/humanoid/stand/stand.h
    tasks/humanoid/tracking/tracking.h
    tasks/humanoid/walk/walk.h
    tasks/manipulation/common.h
    tasks/manipulation/manipulation.h
    tasks/op3/stand.h
    tasks/panda/panda.h
    tasks/particle/particle.h
    tasks/rubik/solve.h
    tasks/shadow_reorient/hand.h
    tasks/quadrotor/quadrotor.h
    tasks/quadruped/quadruped.h
    tasks/swimmer/swimmer.h
    tasks/walker/walker.h
    planners/planner.h
    planners/policy.h
    planners/include.h
    planners/cost_derivatives.h
    planners/model_derivatives.h
    planners/cross_entropy/planner.h
    planners/robust/robust_planner.h
    planners/sample_gradient/planner.h
    planners/sampling/planner.h
    planners/gradient/policy.h
    planners/gradient/settings.h
    planners/ilqg/backward_pass.h
    planners/ilqg/boxqp.h
    planners/ilqg/planner.h
    planners/sampling/policy.h
    planners/gradient/gradient.h
    planners/gradient/planner.h
    planners/gradient/spline_mapping.h
    planners/ilqg/policy.h
    estimators/batch.h
    planners/ilqs/planner.h
    planners/ilqg/settings.h
    planners/rmp/include/core/rmp_space.h
    planners/rmp/include/core/rmp_state.h
    planners/rmp/include/core/rmp_base_policy.h
    planners/rmp/include/core/rmp_policy_value.h
    planners/rmp/include/core/rmp_base_geometry.h
    planners/rmp/include/planner/rmp_planner.h
    planners/rmp/include/planner/rmp_parameters.h
    planners/rmp/include/planner/rmp_trajectory.h
    planners/rmp/include/planner/rmp_base_planner.h
    planners/rmp/include/geometry/rmp_linear_geometry.h
    planners/rmp/include/geometry/rmp_rotated_geometry_3d.h
    planners/rmp/include/geometry/rmp_cylindrical_geometry.h
    planners/rmp/include/policies/rmp_simple_target_policy.h
    planners/rmp/include/eval/rmp_trapezoidal_integrator.h
    planners/rmp/include/util/rmp_vector_range.h
    estimators/estimator.h
    estimators/unscented.h
    estimators/kalman.h
    direct/direct.h
    direct/trajectory.h
    task.h
    norm.h
    spline/spline.h
    direct/model_parameters.h
)

add_library(
  libmjpc STATIC
  states/state.cc
  agent.cc
  trajectory.cc
  utilities.cc
  tasks/tasks.cc
  tasks/acrobot/acrobot.cc
  tasks/allegro/allegro.cc
  tasks/bimanual/insert/insert.cc
  tasks/bimanual/insert/insert.h
  tasks/bimanual/handover/handover.cc
  tasks/bimanual/reorient/reorient.cc
  tasks/cartpole/cartpole.cc
  tasks/fingers/fingers.cc
  tasks/humanoid/stand/stand.cc
  tasks/humanoid/tracking/tracking.cc
  tasks/humanoid/walk/walk.cc
  tasks/manipulation/common.cc
  tasks/manipulation/manipulation.cc
  tasks/op3/stand.cc
  tasks/panda/panda.cc
  tasks/particle/particle.cc
  tasks/rubik/solve.cc
  tasks/shadow_reorient/hand.cc
  tasks/quadrotor/quadrotor.cc
  tasks/quadruped/quadruped.cc
  tasks/swimmer/swimmer.cc
  tasks/walker/walker.cc
  planners/planner.cc
  planners/include.cc
  planners/cost_derivatives.cc
  planners/model_derivatives.cc
  planners/cross_entropy/planner.cc
  planners/robust/robust_planner.cc
  planners/sample_gradient/planner.cc
  planners/sampling/planner.cc
  planners/sampling/policy.cc
  planners/gradient/gradient.cc
  planners/gradient/planner.cc
  planners/gradient/policy.cc
  planners/gradient/spline_mapping.cc
  planners/ilqg/backward_pass.cc
  planners/ilqg/planner.cc
  planners/ilqg/policy.cc
  planners/ilqs/planner.cc
  planners/rmp/src/rmp_planner.cc
  planners/rmp/src/rmp_trajectory.cc
  estimators/batch.cc
  estimators/include.cc
  estimators/include.h
  estimators/kalman.cc
  estimators/unscented.cc
  direct/direct.cc
  direct/model_parameters.cc
  spline/spline.cc
  norm.cc
  task.cc
  ${LIB_MJPC_HEADERS}
)
set_target_properties(libmjpc PROPERTIES OUTPUT_NAME mjpc)
set_target_properties(libmjpc PROPERTIES PUBLIC_HEADER "${LIB_MJPC_HEADERS}")
target_compile_options(libmjpc PUBLIC ${MJPC_COMPILE_OPTIONS})
target_compile_definitions(libmjpc PRIVATE MJSIMULATE_STATIC)
target_link_libraries(
  libmjpc
  absl::any_invocable
  absl::check
  absl::flat_hash_map
  absl::log
  absl::random_random
  absl::span
  glfw
  lodepng
  Eigen3::Eigen
  mujoco::mujoco
  threadpool
  Threads::Threads
)
target_include_directories(libmjpc
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_BINARY_DIR}/..
)

add_executable(
  mjpc
  main.cc
  app.cc
  app.h
  simulate.cc
  simulate.h
  $<TARGET_OBJECTS:mujoco::platform_ui_adapter>
)
target_link_libraries(
  mjpc
  absl::flags
  absl::flags_parse
  absl::random_random
  absl::strings
  libmjpc
  mujoco::mujoco
  mujoco::platform_ui_adapter
  threadpool
  Threads::Threads
)
target_include_directories(mjpc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(mjpc PUBLIC ${MJPC_COMPILE_OPTIONS})
target_link_options(mjpc PRIVATE ${MJPC_LINK_OPTIONS})
target_compile_definitions(mjpc PRIVATE MJSIMULATE_STATIC)
if(APPLE)
  target_sources(mjpc PRIVATE macos_gui.mm)
  target_link_libraries(mjpc "-framework Cocoa")
endif()

add_library(
  libtestspeed STATIC
  testspeed_app.cc
  testspeed.h
  testspeed.cc
)
target_link_libraries(
  libtestspeed
  absl::random_random
  absl::strings
  libmjpc
  mujoco::mujoco
  threadpool
  Threads::Threads
)
target_include_directories(libtestspeed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(
  testspeed
  testspeed_app.cc
)
target_link_libraries(
  testspeed
  libtestspeed
  absl::flags
  absl::flags_parse
)
target_include_directories(testspeed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(testspeed PUBLIC ${MJPC_COMPILE_OPTIONS})
target_link_options(testspeed PRIVATE ${MJPC_LINK_OPTIONS})
target_compile_definitions(testspeed PRIVATE MJSIMULATE_STATIC)

add_subdirectory(tasks)

if(BUILD_TESTING AND MJPC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(MJPC_BUILD_GRPC_SERVICE)
  add_subdirectory(grpc)
endif()

# Install the libraries
if(MJPC_INSTALL_PUBLIC)
  install(
    TARGETS libmjpc
    EXPORT libmjpc
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT runtime
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT runtime
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT dev
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mjpc" COMPONENT dev
  )
endif()
