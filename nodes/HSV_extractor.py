#!/usr/bin/env python

# this is a test script to extract HSv ranges from the tic tac toe image
# 10/1/2021- Unsure if we're keeping it. Need to test.
# 10/6/2021: Use as separate script. Getting board topic isn't working

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
import pyrealsense2 as rs
import time
from math import pi, radians, sqrt, atan
import numpy as np
import matplotlib.pyplot as plt 
import scipy.signal as sig


# Ref
# http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20listener%20%28Python%29
# http://docs.ros.org/en/jade/api/geometry_msgs/html/msg/Transform.html
tf_helper = transformations()
shapeDetect = ShapeDetector()
####


#finding hsv range of target object (pen)
def nothing(x):
    pass


#bridge = CvBridge()
#cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
# # replace data with webcam feed
# # Video Capture doesn't work

# cap = cv_image.copy()

def color_HSV_extract(image):
    print("Your OpenCV version is: " + cv2.__version__)
    
    frame = image.copy()

    # cv2.VideoCapture(0) # replace with Ros Image Topic

    # video capture grabs a frame & then stores it
    # cap.set(3,640) #ours: 640
    # cap.set(4,480) # ours: 480

    # create window named Trackbars
    win_name = 'Trackbars'
    cv2.namedWindow(win_name)


    #Now create 6 tracks to control range of HSV channels
    # Arguments: 
    #      Name of tracker, window name, range, callback function
    # For hue: range is 0-179 
    # Foe S & V: 0-255
    cv2.createTrackbar("Lower-H",win_name,0,179,nothing)
    cv2.createTrackbar("Lower-S",win_name,0,255,nothing)
    cv2.createTrackbar("Lower-V",win_name,0,255,nothing)

    cv2.createTrackbar("Upper-H",win_name,179,179,nothing)  # <-- 179,179 means what?
    '''
    OpenCV docs: https://docs.opencv.org/4.2.0/d7/dfc/group__highgui.html#gaf78d2155d30b728fc413803745b67a9b
    createTrackbar("trackbarname", win_name, value, count, Trackbar Callback)
    value - option pointer to an int variable whose value reflects position of slider
    count - Max position of slider 
    '''
    cv2.createTrackbar("Upper-S",win_name,255,255,nothing)
    cv2.createTrackbar("Upper-V",win_name,255,255,nothing)

    while True:

        # read webcam feed frame by frame
        # ret,frame = cap.read()
        # if not ret:
            # break
        # flip frame horizontally (not required)
        # frame.cv2.flip(frame,1)

        # Convert the BGR image to HSV image.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #Get new values of tracker in real time as user moves slider
        l_h = cv2.getTrackbarPos("Lower-H",win_name)
        l_s = cv2.getTrackbarPos("Lower-S",win_name)
        l_v = cv2.getTrackbarPos("Lower-V",win_name)

        u_h = cv2,getTrackbarPos("Upper-H",win_name) 
        # ^^ as of 10/12/2021: global name 'getTrackbarPos' not defined
        # Issue is with u_h
        # u_h = 179, used this to see further errors
        # need to resolve this issue to get trackbar GUI to appear
        u_s = cv2.getTrackbarPos("Upper-S",win_name)
        u_v = cv2.getTrackbarPos("Upper-V",win_name)

        # Set lower & upper HSB range according to value selected by trackbar
        lower_range = np.array([l_h,l_s,l_v])
        upper_range = np.array([u_h,u_s,u_v])

        # Filter image & get binary mask
        # white represents target color
        mask = cv2.inRange(hsv,lower_range,upper_range)

        # Also visualize real part of target color (Optional)
        res = cv2.bitwise_and(frame,frame,mask-mask)

        # Converting binary mask to 3 channel image
        # so we can stack it with others
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 

        # stack mask, orginal frame & filtered result
        stacked = np.hstack((mask_3,frame,res))

        # show this stacked frame at 40% size
        cv2.imshow(win_name,cv2.resize(stacked, None, fx=0.4,fy=0.4))

        # press ESC then exit program
        key = cv2.waitKey(1)
        if key == 27:
            break


        # if user presses 's' print this array
        # 's' is to save to npy file
        if key == ord('s'):
            thearray =[[l_h, l_s, l_v],[u_h, u_s, u_v]]
            print(thearray)

        # Also save this array as penval.npy
        np.save('hsv_value','thearray')
        break

    # Realse the camera & destroy windows
    # cap.release()
    cv2.destroyAllWindows()

def runner(data):
    """
    Callback function for image subscriber, every frame gets scanned for board and publishes to board_center topic
    (for robot movement) and board tile centers (for game state updates)
    :param camera_data: Camera data input from subscriber
    """
    try:
        # Convert Image to CV2 Frame
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

        # Uncomment below to run HSV extraction code
        # color_HSV_extract(cv_image)

        # Using Image Kernel to detect color
        slide_window(cv_image)


    except rospy.ROSInterruptException:
        exit()
    except KeyboardInterrupt:
        exit()
    except CvBridgeError as e:
        print(e)




def main():
    # Setup Node
    rospy.init_node('HSV_Extractor', anonymous=False)
    rospy.loginfo(">> HSV/Convolution Node Successfully Created")

    # Setup Publishers
    pub_center = rospy.Publisher("ttt_board_origin", TransformStamped, queue_size=20)
    pub_camera_tile_annotation = rospy.Publisher("camera_tile_annotation", Image, queue_size=20)

    # Setup Listeners
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    # bp_callback = board_publisher(pub_center, pub_camera_tile_annotation, tfBuffer)

    # create subscriber to ros Image Topic
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, runner)

    # Auto-Run until launch file is shutdown
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()



if __name__ == '__main__':
        main()


