#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2021, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: LuisC18

## IMPORTS
import pyrealsense2 as rs
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import numpy as np

class ImageProcess(object):
    """
    Class is a collection of Image Process tools for the support of the tictactoe project.
    """

    def __init__(self):
        self.bridge = CvBridge()

        self.color_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_image_callback)
        self.aligned_depth_image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.alighed_depth_image_callback)
        self.intrinsics_sub = rospy.Subscriber("/camera/aligned_depth_to_color/camera_info",CameraInfo, self.intrinsics_callback)

    def color_image_callback(self,color_image_msg):
        """
            ROS subscriber callback that converts the RGB stream sensor_msgs/Image Message to an openCV image and stores it in the self.color_image variable.
        """
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(color_image_msg, "bgr8")

        except CvBridgeError, e:
            print e

    def alighed_depth_image_callback(self, depth_image_msg):
        """
            ROS subscriber callback that converts the depth stream sensor_msgs/Image Message to an openCV image and stores it in the self.color_image variable.
        """
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding="passthrough")
            
        except CvBridgeError, e:
            print e

    def intrinsics_callback(self,camera_info_msg):

        self.camera_intrinsic = np.array(camera_info_msg)

    def depth_at_center_pixel(self):
        depth_array = np.array(self.depth_image, dtype=np.float32)
        center_idx = np.array(depth_array.shape) / 2
        center_pixel = (center_idx[0], center_idx[1])
        center_pixel_depth = depth_array[center_pixel]
        depth_point_in_meters_camera_cords = rs.rs2_deproject_pixel_to_point(self.camera_intrinsic, center_pixel, center_pixel_depth)
        print(depth_point_in_meters_camera_cords)

    
    
    



    # def detectBoard_contours(image):
    #     """
    #         Tictactoe function that finds the physical board. Utilizes ShapeDetector class functions.
    #         @param image: image parameter the function tries to find the board on.
    #         @return scaledCenter: (x ,y) values in meters of the board center relative to the center of the camera frame.
    #         @return boardImage: image with drawn orientation axes and board location.
    #         @return tf_camera2board: transformation matrix of board to camera frame of reference.
    #         """





def main():
    rospy.init_node('image_processor', anonymous=False)
    IP = ImageProcess()
    IP.depth_at_center_pixel()
    rospy.spin()

if __name__ == '__main__':
    main()