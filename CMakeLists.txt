cmake_minimum_required(VERSION 3.1.2)
project(classification)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  message_generation
  geometry_msgs
  sensor_msgs
  rospix
  cmake_modules
  )

add_message_files(DIRECTORY msg FILES

  Pixel.msg
  Cluster.msg
  ClusterClass.msg
  ClusterCounts.msg
  ProcessedImage.msg
  )

generate_messages(DEPENDENCIES

  std_msgs
  geometry_msgs
  sensor_msgs
  )

catkin_package(
  # INCLUDE_DIRS include
  CATKIN_DEPENDS rospy std_msgs rospix message_runtime
  DEPENDS message_runtime
  )

include_directories(
  ${catkin_INCLUDE_DIRS}
  )
