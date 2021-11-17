#!/usr/bin/env python

# this is a test script to use the kernel method to find color positions in an image
#
#####################################################
#####################################################
## IMPORTS
import sys
import os

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
import imutils

tf_helper = transformations()
shapeDetect = ShapeDetector()
# colorFinder = ColorFinder() # 11/8: not working


# Add game_scripts directory to the python-modules path options to allow importing other python scripts
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '//home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/game_scripts')
ttt_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_2_game_scripts = ttt_pkg + '/game_scripts'
sys.path.insert(1, path_2_game_scripts)



### Main Function
# Creating kernel
kernel_size = 25
kernel_blue = np.zeros((kernel_size, kernel_size))  # creating an empty array
# 11/11: increased size from 5x5

frame = cv2.imread('twistCorrectedColoredSquares_Color.png')
# 11/15: Image stored on desktop b/c using Scott Labs computer

# Uncomment below to use RGB values in Kernel
kernel_blue = np.append(kernel_blue, [0, 0, 255])
kernel_blue = np.asanyarray(kernel_blue, np.float32)

anchor = (kernel_size / 2.0, kernel_size / 2.0)

blue_heatmap = cv2.filter2D(frame, -3, kernel_blue, anchor)

plt.figure(1)
plt.imshow(frame)
plt.figure(2)
plt.imshow(blue_heatmap)
plt.show()








