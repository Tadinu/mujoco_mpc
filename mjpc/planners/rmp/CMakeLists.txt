find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS " -Wall -Wextra")
set(CMAKE_BUILD_TYPE Release)

set(SOURCES
    src/rmp_planner.cc
    src/rmp_trajectory.cc)

set(TESTSOURCES
    test/tester.cc
    test/parser.cc)

# Main executable
add_executable(rmp_test
        ${SOURCES}
        ${TESTSOURCES}
        test/run_test.cc)
target_link_libraries(rmp_test
        Eigen3::Eigen)

