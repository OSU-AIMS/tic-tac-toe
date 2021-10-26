#!/usr/bin/env python

# this is a test script to use the kernel method to find color positions in an image
#
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


def kernel_color_detect(image):
    '''
    purpose: slides an array (nxn pixels) across an image to find color
    - obtain heat map of the color
    - find pixel location
    - get transform
    '''
    # Uncomment below for at home testing with webcam
    # cap = cv2.VideoCapture(0)
    # cap.set(3,1280)
    # cap.set(4,720) 

    # (RGB)
    # Red: (255,0,0)
    # Green: (0,255,0)
    # Blue: (0,0,255)
    print("Your OpenCV version is: " + cv2.__version__)  
    
    # Uncomment below when using camera feed
    frame = image.copy()
    # rows = len(frame)
    # cols = len(frame[0])
    # print(rows) # 480
    # print(cols) # 640

    # Uncomment below when using image
    # test_image= os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/sample_content/sample_images/CorrectedColoredSquares_Color.png'
    # frame = cv2.imread(test_image)
    
    # Uncomment for Debuggging Frame
    # print(frame)
    # size = frame.shape

    # print(size)


    # make matrix size 50x50 with rgb values inside
    # red = [255,0,0]
    # green = [0,255,0]
    # blue = [0,0,255]

    # Ref: https://www.geeksforgeeks.org/image-filtering-using-convolution-in-opencv/amp/
    # Plan as of 10/20/21: use read image into kernel matrix then perform convolutions 
    # GeeksforGeeks use Python 3 (we have Python 2.7)

    # Obtain directory for blue, green, red square crops
    blue_square_crop = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/sample_content/sample_images/blue_square_crop.png'
    green_square_crop = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/sample_content/sample_images/green_square_crop.png'
    red_square_crop = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/sample_content/sample_images/red_square_crop.png'

    # Numpy: uses (y,x) / OpenCv (x,y)
    blue_square = cv2.imread(blue_square_crop)
    kernel_blue = np.zeros((51,51))  # creating an empty array
    # Uncomment below to use RGB values
    kernel_blue = np.append(kernel_blue,[255,0,0])
    # Uncomment below to use image
    # kernel_blue = np.append(blue_square)
    
    print('kernel_blue:',kernel_blue) # --> 10/26: image being read
    kernel_blue = np.asanyarray(kernel_blue,np.float32)
    # ^ Kernel values need to be floating-point numbers
    # numpy.asanyarray({insert Kernel variable}, np.float32)
    
    '''
    ISSUE: 10/26
    blue-heat map shows all white except for a few splotches that don't match
    '''


    # print('Kerenel')
    # print(kernel_blue) # Also not finding the Kernel
    # Kernel size must by n x n where n- odd numbers
    # blue, green, and red square crops are 55 x 55 pixels




    blue_heatmap = cv2.filter2D(frame,-1,kernel_blue)
    print('Blue HeatMap: ',blue_heatmap)
    # 10/26: outputs white plot with yellow & blue splotches
    
    # opencv docs on filter2D:
    # https://docs.opencv.org/4.2.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    '''
    void cv::filter2D(
        InputArray src, 
        OutputArray dst, 
        int ddepth, 
        InputArray kernel,
        Point anchor = Point(-1,-1),
        double delta = 0,
        int borderType = BORDER_DEFAULT 
    )   
    '''
    # cv2.imshow('Original Image',frame)
    # cv2.waitKey(0)
    plt.imshow(blue_heatmap)
    plt.show()

    cv2.destroyAllWindows()

    # Finding Blue Square - origin
    # window = win_im.dot(blue)
    # print(window)

    # frame = image.copy()      # frame to slide window over

    # # conv = cv2.convolve(frame,window)
    # # OpenCV docs: https://docs.opencv.org/4.2.0/d4/d25/classcv_1_1cuda_1_1Convolution.html
    # conv = cv2.filter2D(frame,-1,win_im)
    # cv2.imshow("Convolution",conv)
    cv2.waitKey(0)

    # Uncomment below when using webcam
    # cap.release()

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

        # Using Image Kernel to detect color
        kernel_color_detect(cv_image)


    except rospy.ROSInterruptException:
        exit()
    except KeyboardInterrupt:
        exit()
    except CvBridgeError as e:
        print(e)



def main()
    # Setup Node
    rospy.init_node('Kernel Color Detect', anonymous=False)
    rospy.loginfo(">> Kernel Color Detect Node Successfully Created")

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

