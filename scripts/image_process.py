#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2021, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: LuisC18

## IMPORTS
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np

class ImageProcess(object):
    """
    Class is a collection of Image Process tools for the support of the tictactoe project.
    """

    def __init__(self):

        self.color_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_image_callback)
        self.aligned_depth_image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.alighed_depth_image_callback)

    def color_image_callback(self,msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg.data, "bgr8")

    def alighed_depth_image_callback(self, msg):

        try:
            depth_image = bridge.imgmsg_to_cv2(msg.data, desired_encoding="passthrough")
            depth_array = np.array(depth_image, dtype=np.float32)
            center_idx = np.array(depth_array.shape) / 2
            print('center depth:', depth_array[center_idx[0], center_idx[1]])

        except CvBridgeError, e:
            print e


    def intrinsics(self):

    def detectBoard_contours(image):
        """
            Tictactoe function that finds the physical board. Utilizes ShapeDetector class functions.
            @param image: image parameter the function tries to find the board on.
            @return scaledCenter: (x ,y) values in meters of the board center relative to the center of the camera frame.
            @return boardImage: image with drawn orientation axes and board location.
            @return tf_camera2board: transformation matrix of board to camera frame of reference.
            """



if __name__ == '__main__':
    rospy.init_node('image_processor', anonymous=False)
    ImageProcess()
    rospy.spin()