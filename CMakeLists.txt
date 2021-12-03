cmake_minimum_required(VERSION 3.0.2)
project(tic_tac_toe)

find_package(catkin REQUIRED
  COMPONENTS
    roscpp
    rospy
    std_msgs
    message_generation
)

add_service_files(
  FILES
  LoadCameraInfo.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package()

install(
    PROGRAMS nodes_new/camera_info_server.py
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
    DIRECTORY edge_scripts color_scripts volume_scripts nodes nodes_new launch urdf srv
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
