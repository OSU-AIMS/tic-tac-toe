#!/usr/bin/env python3
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2021, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: acbuynak

# Imports
import sys
import rospy
import pyrealsense2 as pyirs
from tic_tac_toe.srv import LoadCameraInfo
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo


class vision_server():
    def __init__(self, camera_info_topic):
        self.camera_info_topic = camera_info_topic
        self.callback_loadCameraInfo()
        print("insitial setup")

    def set_camera_parameters(self, cameraInfo):
        rospy.set_param("intrinsics/width",     cameraInfo.width)
        rospy.set_param("intrinsics/height",    cameraInfo.height)
        rospy.set_param("intrinsics/ppx",       cameraInfo.K[2])
        rospy.set_param("intrinsics/ppy",       cameraInfo.K[5])
        rospy.set_param("intrinsics/fx",        cameraInfo.K[0])
        rospy.set_param("intrinsics/fy",        cameraInfo.K[4])
        rospy.set_param("intrinsics/model",     cameraInfo.distortion_model)
        rospy.set_param("intrinsics/coeffs",    [i for i in cameraInfo.D])

        rospy.loginfo("Sucessfully loaded camera info from topic: %s", self.camera_info_topic)

    def callback_loadCameraInfo(self, empty=""):
        # Disregard empty passed to callback function. Required to enable python callback.
        msg = rospy.wait_for_message(self.camera_info_topic, CameraInfo, timeout=None)
        self.set_camera_parameters(msg)
        return True


def main(topic_name):
    """
    Setup vision server to perform below..
    - Lookup camera info topic and store in parameter server.
    - Setup ROS Service to allow reloading camera info parameters
      Service will retain node's namespace.
    """

    # Setup & Initial Camera Data Load
    name = rospy.init_node('vision_server', anonymous=False)
    server = vision_server(topic_name)

    # Setup Service
    rospy.Service('reload_camera_info', LoadCameraInfo, server.callback_loadCameraInfo)

    # Run
    rospy.spin()


if __name__ == "__main__":

    # Check Input Arguments
    if len(sys.argv) < 1:
        print("usage: camera_info_server.py str(camera_info_topic)")

    #Launch Node
    else:
        try:
            main(sys.argv[1])
        except rospy.ROSInterruptException:
            pass
