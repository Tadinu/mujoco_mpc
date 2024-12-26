# Copyright 2021 DeepMind Technologies Limited
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

cmake_minimum_required(VERSION 3.16)

# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Default to GLVND if available.
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

# This line has to appear before 'PROJECT' in order to be able to disable incremental linking
set(MSVC_INCREMENTAL_DEFAULT ON)

set(MUJOCO_DEP_VERSION_lodepng
    b4ed2cd7ecf61d29076169b49199371456d4f90b
    CACHE STRING "Version of `lodepng` to be fetched."
)

project(
  mjapp
  DESCRIPTION "MuJoCo App binaries"
  HOMEPAGE_URL "https://mujoco.org"
)

enable_language(C)
enable_language(CXX)
if(APPLE)
  enable_language(OBJC)
  enable_language(OBJCXX)
endif()

option(SIMULATE_BUILD_EXECUTABLE "Build the simulate executable binary." ON)
option(SIMULATE_GLFW_DYNAMIC_SYMBOLS "Whether to resolve GLFW symbols dynamically." OFF)

# Check if we are building as standalone project.
set(SIMULATE_STANDALONE ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../cmake")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../cmake")

if(SIMULATE_STANDALONE)
  include(SimulateOptions)
else()
  enforce_mujoco_macosx_min_version()
endif()
include(SimulateDependencies)

# Fetch lodepng dependency.
if(NOT TARGET lodepng)
  FetchContent_Declare(
    lodepng
    GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
    GIT_TAG ${MUJOCO_DEP_VERSION_lodepng}
  )

  FetchContent_GetProperties(lodepng)
  if(NOT lodepng_POPULATED)
    FetchContent_Populate(lodepng)
    # This is not a CMake project.
    set(LODEPNG_SRCS ${lodepng_SOURCE_DIR}/lodepng.cpp)
    set(LODEPNG_HEADERS ${lodepng_SOURCE_DIR}/lodepng.h)
    add_library(lodepng STATIC ${LODEPNG_HEADERS} ${LODEPNG_SRCS})
    target_compile_options(lodepng PRIVATE ${MUJOCO_MACOS_COMPILE_OPTIONS})
    target_link_options(lodepng PRIVATE ${MUJOCO_MACOS_LINK_OPTIONS})
    target_include_directories(lodepng PUBLIC ${lodepng_SOURCE_DIR})
  endif()
endif()

# Simulate library
add_library(mjapp_platform_ui_adapter OBJECT)
target_sources(
  mjapp_platform_ui_adapter
  PUBLIC glfw_adapter.h glfw_dispatch.h platform_ui_adapter.h
  PRIVATE glfw_adapter.cc glfw_dispatch.cc platform_ui_adapter.cc
)
target_compile_options(mjapp_platform_ui_adapter PRIVATE ${MJPC_COMPILE_OPTIONS})
if(APPLE)
  target_sources(mjapp_platform_ui_adapter PUBLIC glfw_corevideo.h PRIVATE glfw_corevideo.mm)
  target_link_libraries(mjapp_platform_ui_adapter PUBLIC "-framework CoreVideo")
endif()
target_include_directories(
  mjapp_platform_ui_adapter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                             $<TARGET_PROPERTY:glfw,INTERFACE_INCLUDE_DIRECTORIES>
)
target_link_libraries(mjapp_platform_ui_adapter PUBLIC mujoco::mujoco)
if(SIMULATE_GLFW_DYNAMIC_SYMBOLS)
  target_compile_definitions(mjapp_platform_ui_adapter PUBLIC mjGLFW_DYNAMIC_SYMBOLS)
endif()
add_library(mujoco::mjapp_platform_ui_adapter ALIAS mjapp_platform_ui_adapter)

add_library(libmjapp_simulate STATIC $<TARGET_OBJECTS:mjapp_platform_ui_adapter>)
set_target_properties(libmjapp_simulate PROPERTIES OUTPUT_NAME mjapp)
add_library(mujoco::libmjapp_simulate ALIAS libmjapp_simulate)
target_sources(
  libmjapp_simulate
  PUBLIC sim_base.h sim.h robot_model.h
  PRIVATE sim_base.cc sim.cc array_safety.h
)

target_include_directories(libmjapp_simulate
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${casadi_INCLUDE_DIR}
  ${DQ_DIR}/include
  ${DM_ROBOTICS_DIR}/controllers/lsqp/include
  ${DM_ROBOTICS_DIR}/least_squares_qp/core/include
  ${DM_ROBOTICS_DIR}/least_squares_qp/common/include
  ${DM_ROBOTICS_DIR}/mujoco/include
  ${DM_ROBOTICS_DIR}/support/include
  ${manif_INCLUDE_DIRS}
)
target_compile_options(libmjapp_simulate PRIVATE ${MJPC_COMPILE_OPTIONS})
target_link_libraries(libmjapp_simulate PUBLIC
  lodepng mujoco::platform_ui_adapter mujoco::mujoco
  #absl::any_invocable
  #absl::check
  #absl::flat_hash_map
  #absl::log
  absl::random_random
  absl::span
  absl::strings
  absl::status
  absl::statusor
  Eigen3::Eigen
  casadi
  ${DQ_DIR}/lib/libdqrobotics.so
  ${DM_ROBOTICS_BUILD_DIR}/controllers/lsqp/libdmr_controllers_lsqp.so
  ${DM_ROBOTICS_BUILD_DIR}/mujoco/libdmr_mujoco.so
  ${DM_ROBOTICS_BUILD_DIR}/least_squares_qp/core/libdmr_lsqp_core.so
  ${DM_ROBOTICS_BUILD_DIR}/least_squares_qp/common/libdmr_lsqp_common.so
  qp_solver_collection::QpSolverCollection
  ${QP_SOLVER_3RD_LIBS}
)
target_link_options(libmjapp_simulate PRIVATE ${MJPC_LINK_OPTIONS})

if(APPLE)
  target_sources(libmjapp_simulate PRIVATE macos_gui.mm)
  target_link_libraries(libmjapp_simulate PUBLIC "-framework Cocoa")
endif()

# Build simulate executable
if(SIMULATE_BUILD_EXECUTABLE)
  if(APPLE)
    set(SIMULATE_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../dist/mujoco.icns)
  elseif(WIN32)
    set(SIMULATE_RESOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../dist/simulate.rc)
  else()
    set(SIMULATE_RESOURCE_FILES "")
  endif()

  add_executable(mjapp main.cc array_safety.h ${SIMULATE_RESOURCE_FILES})
  target_compile_options(mjapp PUBLIC ${MJPC_COMPILE_OPTIONS})
  if(WIN32)
    add_custom_command(
      TARGET mjapp
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/../dist/mujoco.ico
              ${CMAKE_CURRENT_SOURCE_DIR}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E rm ${CMAKE_CURRENT_SOURCE_DIR}/mujoco.ico
    )
  endif()

  target_include_directories(mjapp
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
  )

  target_link_libraries(
    mjapp
    libmjapp_simulate
    mujoco::mujoco
    libmjpc
    glfw
    Threads::Threads
    lodepng
    absl::span
    absl::hash
    absl::btree
    absl::hashtablez_sampler
    absl::strings
    absl::status
    absl::statusor
  )

  target_link_options(mjapp PRIVATE ${MJPC_LINK_OPTIONS})
endif()
