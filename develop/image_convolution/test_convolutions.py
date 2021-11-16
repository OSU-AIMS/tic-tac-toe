#!/usr/bin/env python

# Script to test smaller kernel convolutions

import rospy
import tf2_ros
import cv2

# ROS Data Types
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

# Custom Tools
    # from Realsense_tools import *
from transformations import *
from shape_detector import *
from cv_bridge import CvBridge, CvBridgeError
# from color_finder import *  # 11/8: not working

# System Tools
import pyrealsense2 as rs
import time
from math import pi, radians, sqrt, atan, ceil
import numpy as np
import matplotlib.pyplot as plt


def kernel_runner(image):
    # Create kernel (format - "BGR")
    kernel_size  = 3
    kernel_b     = 255*np.ones((kernel_size, kernel_size, 1), dtype='uint8')
    kernel_gr    = np.zeros((kernel_size, kernel_size, 2), dtype='uint8')
    kernel       = np.dstack((kernel_b,kernel_gr))

    output = cv2.filter2D(image,kernel,-1)

    # Next: 
    # Apply convolution of 3x3


    '''
    filter2D parameters:
        InputArray src, 
        OutputArray dst, 
        int ddepth, 
        InputArray kernel,
        Point anchor = Point(-1,-1),
        double delta = 0,
        int borderType = BORDER_DEFAULT   
    # opencv docs on filter2D:
    # https://docs.opencv.org/4.2.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    '''




def main():
    image = cv2.imread('test_grid.tif')