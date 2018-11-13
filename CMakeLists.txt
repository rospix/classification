cmake_minimum_required(VERSION 2.8.3)
project(rospix_classification)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

set(CMAKE_CXX_FLAGS "-std=c++17 -fno-diagnostics-color")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  message_generation
  geometry_msgs
  sensor_msgs
  nodelet
  rospix
  cmake_modules
)

# messages for MPC_TRACKER
add_message_files(DIRECTORY msg FILES

  Pixel.msg
  Cluster.msg
  ClusterClass.msg
  ClusterCounts.msg
  ProcessedImage.msg
  )

## Generate added messages and services with any dependencies listed here
generate_messages(DEPENDENCIES

  std_msgs
  geometry_msgs
  sensor_msgs
  )

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gdax_interface
   CATKIN_DEPENDS nodelet rospy std_msgs rospix message_runtime
   DEPENDS message_runtime
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(${catkin_INCLUDE_DIRS})
