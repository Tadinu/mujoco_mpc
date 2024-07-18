# https://github.com/ORB-HD/URDF_Parser
CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(urdf_parser)

FIND_PACKAGE(Boost REQUIRED)

SET(CMAKE_CXX_STANDARD 17)

INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

SET(URDF_SRCS
  common.cpp
  joint.cpp
  geometry.cpp
  link.cpp
  model.cpp
  tinyxml.cpp
  tinyxmlerror.cpp
  tinyxmlparser.cpp
)

ADD_LIBRARY(urdf_parser STATIC
  ${URDF_SRCS}
)

SET(URDFPARSER_LIB urdfparser)

SET_PROPERTY(TARGET urdfparser PROPERTY POSITION_INDEPENDENT_CODE ON)

