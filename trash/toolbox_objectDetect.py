#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2021, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: MohammadKhan-3

## IMPORTS
import pyrealsense2 as rs
import rospy
import numpy as np
from math import pi, radians, sqrt, atan, ceil


def object_detect(image):
    # Add MatchTemplate stuff once finished
    # 12/2: update matchTemplate to have RGB values as kernels instead of images as kernels so it's more generally applicable
    # output centers of bounding boxes (Tuples)






def main():
    """
    Purpose: continuously output position based on kernel detection
    """
    # Setup Node 
    rospy.init_node('object_detect', anonymous=False)
    print('object_detect node successfully created!')

    # Setup Listener for /camera/color/image_raw
    image = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=None)

    # Output board position
    # Create Publisher    # publish to Image topic?
    pub_boardPOS = rospy.Publisher("board_position",Image,queue_size=20)
    
    positions = object_detect(image) # Add pub_boardPOS into parameters of object_detect
    pub_boardPOS.publish(positions)


    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
     try:
        main()
    except rospy.ROSInterruptException:
        pass