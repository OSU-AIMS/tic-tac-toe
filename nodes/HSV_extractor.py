#!/usr/bin/env python

# this is a test script to extract HSv ranges from the tic tac toe image
# 10/1/2021- Unsure if we're keeping it. Need to test.

# Script from:
 # https://medium.com/programming-fever/how-to-find-hsv-range-of-an-object-for-computer-vision-applications-254a8eb039fc

#####################################################
#####################################################
## IMPORTS
import sys
import os

# Add game_scripts directory to the python-modules path options to allow importing other python scripts
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '//home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/game_scripts')
ttt_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_2_game_scripts = ttt_pkg + '/game_scripts'
sys.path.insert(1, path_2_game_scripts)

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

# System Tools
import time
from math import pi, radians, sqrt, atan
import numpy as np


# Ref
# http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20listener%20%28Python%29
# http://docs.ros.org/en/jade/api/geometry_msgs/html/msg/Transform.html
tf_helper = transformations()
shapeDetect = ShapeDetector()
####


#finding hsv range of target object (pen)
def nothing(x):
	pass

print("Your OpenCV version is: " + cv2.__version__)


# initialize webcam feed
# Convert Image to CV2 Frame
bridge = CvBridge()
cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
# replace data with image
cap = cv_image.copy()
# cap = cv2.VideoCapture(0)
cap.set(3,1280) #ours: 640
cap.set(4,720) # ours: 480

# create window named Trackbars
cv2.namedWindow('Trackbars for HSV')

#Now create 6 tracks to control range of HSV channels
# Arguments: 
#      Name of tracker, window name, range, callback function
# For hue: range is 0-179 
# Foe S & V: 0-255
cv2.createTrackbar("Lower-H","Trackbars",0,179,nothing)
cv2.createTrackbar("Lower-S","Trackbars",0,255,nothing)
cv2.createTrackbar("Lower-V","Trackbars",0,255,nothing)

cv2.createTrackbar("Upper-H","Trackbars",179,179,nothing)
cv2.createTrackbar("Upper-S","Trackbars",255,255,nothing)
cv2.createTrackbar("Upper-V","Trackbars",255,255,nothing)

while True:

	# read webcame feed frame by frame
	ret,frame = cap.read()
	if not ret:
		break
	# flip frame horizontally (not required)
	# frame.cv2.flip(frame,1)

	#Get new values of tracker in real time as user moves slider
	l_h = cv2.getTrackbarPos("Lower-H","Trackbars")
	l_s = cv2.getTrackbarPos("Lower-S","Trackbars")
	l_v = cv2.getTrackbarPos("Lower-V","Trackbars")

	u_h = cv2,getTrackbarPos("Upper-H","Trackbars")
	u_s = cv2.getTrackbarPos("Upper-S","Trackbars")
	u_v = cv2.getTrackbarPos("Upper-V","Trackbars")

	# Set lower & upper HSB range according to value selected by trackbar
	lower_range = np.array([l_h,l_s,l_v])
	upper_range = np.array([u_h,u_s,u_v])

	# Filter image & get binary mask
	# white represents target color
	mask = cv2.inRange(hsv,lower_range,upper_range)

	# Also visualize real part of target color (Optional)
	# res = cv2.bitwise_and(frame,frame,mask-mask)

	# Converting binary mask to 3 channel image
	# so we can stack it with others
	mask_3 = cv2.cvtColor(mask, cv2.GRAY2BGR)

	# stack mask, orginal frame & filtered result
	stacked = np.hstack((mask_3,frame,res))

	# show this stacked frame at 40% size
	cv2.imshow('Trackbars',cv2.resize(stacked, None, fx=0.4,fy=0.4))

	# press ESC then exit program
	key = cv2.waitKey(1)
	if key == 27:
		break


	# if user presses 's' print this array
	# 's' is to save to npy file
	if key == ord('s'):
		thearray =[[l_h,l_s,l_v],[u_h,u_s,u_v]]
		print(thearray)

	# Also save this array as penval.npy
	np.save('hsv_value','thearray')
	break

# Realsense camera & destroy windows

cap.release()
cv2.destroyAllWindows()
