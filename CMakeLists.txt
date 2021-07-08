cmake_minimum_required(VERSION 2.8.3)

project(tic_tac_toe)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  )

catkin_package()

install(DIRECTORY edge_scripts volume_scripts
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
