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
# import scipy.signal as sig


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
    # cv2.imshow('Frame',frame)
    # cv2.waitKey(0)

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
    
    # Uncomment: Checking to see if original image is flipped
    # cv2.imshow('Blue Square',blue_square)
    # cv2.waitKey(0)
    # 10/28: Blue square is not flipped


    kernel_blue = np.zeros((51,51))  # creating an empty array
    # Uncomment below to use RGB values
    kernel_blue = np.append(kernel_blue,[0,0,255])
    # Uncomment below to use image
    # kernel_blue = np.append(kernel_blue,blue_square)
    
    print('kernel_blue:',kernel_blue) # --> 10/26: image being read
    kernel_blue = np.asanyarray(kernel_blue,np.float32)

    # kernel_blue = cv2.flip(kernel_blue,-1) # flipping kernel horizontally and vertically
    # can't assume kernel to be symmetric matrix, need to flip it before passing it to filter2D
    # negative number: flips about both axes
    #  0: flip around x-axis
    # postive number: flip around y-axis
    # docs on cv2.flip: https://docs.opencv.org/4.5.3/d2/de8/group__core__array.html#gaca7be533e3dac7feb70fc60635adf441
 
    '''
    ISSUE: 10/28:
    - how to obtain pixel location:
    potential help: https://stackoverflow.com/questions/59599568/find-sub-pixel-maximum-on-a-2d-array

    '''


    # print('Kerenel')
    # print(kernel_blue) # Also not finding the Kernel
    # Kernel size must by n x n where n- odd numbers
    # blue, green, and red square crops are 55 x 55 pixels
    blue_heatmap = cv2.filter2D(frame,-3,kernel_blue)
    print('Blue HeatMap: ',blue_heatmap)
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

# use colored square detection to detect contour in heatmap, get position, orientation
def detectBoard_coloredSquares(image):
    # purpose: recognize orientation of board based on 3 colored equares
    # @param: imageFrame - frame pulled from realsense camera

    print("Your OpenCV version is: " + cv2.__version__)
    # import image frame with colored squares

    # imageframe = cv2.imread('/sample_content/sample_images/twistCorrectedColoredSquares_Color.png') 

   
    # Next Recognize Location & Centers 
    # Python code for Multiple Color Detection (from: https://www.geeksforgeeks.org/multiple-color-detection-in-real-time-using-python-opencv/)
    # use bounding boxes to get square centers

    # Capturing video through webcam
    # webcam = cv2.VideoCapture(0)
    #^^ test this also

    # Start a while loop
    # while(1):     
    # Reading the video from the
    # webcam in image frames
    
    #_, imageFrame = webcam.read()

    imageFrame = image.copy()
    # scale = .895 / 1280         # res: (1280x730)
    scale = .65/640

    cv2.imshow("Image Frame",imageFrame)
    cv2.waitKey(3)
    # used for debugging purposes

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV frame',hsvFrame)
    cv2.waitKey(3)

    '''
    from https://stackoverflow.com/questions/36817133/identifying-the-range-of-a-color-in-hsv-using-opencv/51686953
    color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
              'white': [[180, 18, 255], [0, 0, 231]],
              'red1': [[180, 255, 255], [159, 50, 70]],
              'red2': [[9, 255, 255], [0, 50, 70]],
              'green': [[89, 255, 255], [36, 50, 70]],
              'blue': [[128, 255, 255], [90, 50, 70]],
              'yellow': [[35, 255, 255], [25, 50, 70]],
              'purple': [[158, 255, 255], [129, 50, 70]],
              'orange': [[24, 255, 255], [10, 50, 70]],
              'gray': [[180, 18, 230], [0, 0, 40]]}
    '''

    # Set range for red color and define mask
    # red_lower = np.array([136, 87, 111], np.uint8)
    # red_upper = np.array([180, 255, 255], np.uint8)

    # are thresholds below rbg (not this) or rgb(not this) or bgr(not this) or gbr(not this)
    # psych: It's converting BGR to HSV

    # Set range for red color and define mask
    red_lower = np.array([170, 100, 100], np.uint8)
    # [0, 190, 210]
    # [0, 190, 220]
    red_upper = np.array([180, 255, 255], np.uint8)
    # [2, 220, 255]
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for green color and define mask
    # values are V,S,H b/c color values are in BGR
    # green_lower = np.array([25, 52, 72], np.uint8)
    # green_upper = np.array([102, 255, 255], np.uint8)
    green_lower = np.array([80, 230, 130], np.uint8)
    # [80, 230, 150]
    green_upper = np.array([84, 255, 200], np.uint8)
    # [84, 255, 215]
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Set range for blue color and define mask
    # blue_lower = np.array([94, 80, 2], np.uint8)
    # blue_upper = np.array([120, 255, 255], np.uint8)
    blue_lower = np.array([100, 254, 140], np.uint8)
    # [100, 254, 140]
    blue_upper = np.array([104, 255, 185], np.uint8)
    # [104, 255, 175]
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
    
    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")
    
    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame, mask = red_mask)
    
    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame, mask = green_mask)
    
    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask = blue_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300): # assuming area is in pixels
            x, y, w, h = cv2.boundingRect(contour)
            center_R = [w/2+x,h/2+y]
            # imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # ^^ Uncomment if you want bounding boxes to appear
            
            # drawing rect around blue sqaure
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(imageFrame,[box],0,(0,0,0),2)
            cv2.putText(imageFrame, "Red Color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))    

    # Creating contour to track green color
    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            center_G = [w/2+x,h/2+y]
            #imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # ^^ Uncomment if you want bounding boxes to appear


            # drawing rect around green sqaure
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(imageFrame,[box],0,(0,0,0),2)

            cv2.putText(imageFrame, "Green Color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300): # original : area>300
            x, y, w, h = cv2.boundingRect(contour)
            center_B = [w/2+x,h/2+y]
            # imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # ^^ Uncomment if you want bounding boxes to appear

            # drawing rect around blue sqaure
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(imageFrame,[box],0,(0,0,0),2)
            
            cv2.putText(imageFrame, "Blue Color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
            
    
    # Used for debugging and checking values
    print('Center of Blue Bounding Box: ',center_B)
    print('Center of Green Bounding Box: ',center_G)
    print('Center of Red Bounding Box: ',center_R)



    # get angle from red to green & use that as the orientation
        # X-vector: Blue --> Red
        # Y-vector: Blue --> Green


    # then use DrawAxis function to draw x(blue - red) & y axis (blue to green) 
    shapeDetect.drawAxis(imageFrame, center_B, center_G, (9, 195, 33), 1) # Y-axis GREEN
    shapeDetect.drawAxis(imageFrame, center_B, center_R, (104,104, 255), 1) # X-Axis RED
    #angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    # Program Termination
    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
    # cv2.waitKey(3)
    # if cv2.waitKey(3) & 0xFF == ord('q'):
    #     cap.release()
    #     cv2.destroyAllWindows()
        # break

    # Use centers 
    # boardImage, _ , _ = shapeDetect.detectSquare(image.copy(), area=90000)
    # removed boardPoints and boardCenter => this function should find boardPoints and boardCenter
    # original: area=90000
    # 1st change- area = 5000

    # Get distance from each center to get board center
    x_axis = [center_R[0] - center_B[0], center_R[1] - center_B[1]] 
    y_axis = [center_G[0] - center_B[0], center_G[1] - center_B[1]]
    # print("X-axis: ", x_axis)
    # print("Y-axis: ", y_axis)

    # Convert board center pixel values to meters (and move origin to center of image)
    scaledCenter = [0,0]
    scaledCenter[0] = ((x_axis[0]+x_axis[1])/2 - 320) * scale
    scaledCenter[1] = ((y_axis[0]+y_axis[1])/2 - 240) * scale 
    print("Scaled Center (m): ",scaledCenter)
    # scaledCenter = [x_axis[0]/2,y_axis[0]/2]

    # vertical line used as reference line
    # vertical = cv2.line(imageFrame,[437,4], [437,713],(0,0,0), 1)
    # vertical = [437,713-4]
    # 
    # shapeDetect.drawAxis(imageFrame,[437,4],[437,713],(0,0,0),1)
    # cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)

    # vector from red square (bottom right) to green square (top left)
    angle = math.degrees(math.atan2(center_R[1] - center_B[1], center_R[0] - center_B[0])) # in degrees 
    print("Angle(deg): ", angle)


    # Define 3x1 array of board translation (x, y, z) in meters
    boardTranslation = np.array(
        [[scaledCenter[0]], [scaledCenter[1]], [0.655]])  ## depth of the table is .655 m from camera


    # convert angle to a rotation matrix with rotation about z-axis
    boardRotationMatrix = np.array([[math.cos(radians(angle)), -math.sin(radians(angle)), 0],
                                    [math.sin(radians(angle)), math.cos(radians(angle)), 0],
                                    [0, 0, 1]])

    # Transformation (board from imageFrame to camera)
    tf_camera2board = tf_helper.generateTransMatrix(boardRotationMatrix, boardTranslation)


    # Assemble Annotated Image for Output Image
    boardImage = image.copy()
    shapeDetect.drawAxis(boardImage, center_B, center_G, (9, 195, 33), 1)   # Y-axis GREEN
    shapeDetect.drawAxis(boardImage, center_B, center_R, (104,104, 255), 1) # X-Axis RED
    cv2.putText(boardImage,
        "Board Rotation Technique: RGB Corner Squares",
        org = (20, 20),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.5,
        color = (0,0,0),
        thickness = 2,
        lineType = cv2.LINE_AA,
        bottomLeftOrigin = False)
    text_angle = "Angle(deg): " + str(int(angle))
    cv2.putText(boardImage,
        text_angle,
        org = (20, 40),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.5,
        color = (0,0,0),
        thickness = 2,
        lineType = cv2.LINE_AA,
        bottomLeftOrigin = False)


    # Returns Center Location, CV2 image annotated with vectors used in analysis, TF
    return scaledCenter, boardImage ,tf_camera2board 

def runner(data):
    """
    Callback function for image subscriber, every frame gets scanned for board and publishes to board_center topic
    (for robot movement) and board tile centers (for game state updates)
    :param camera_data: Camera data input from subscriber
    """
    try:
        # Convert Image to CV2 Frame
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(data, "rgb8") 
        # OpenCV:BGR / RealSense: RGB / RGB: to get proper colors --> also filps colors in frame

        # Using Image Kernel to detect color
        kernel_color_detect(cv_image)


    except rospy.ROSInterruptException:
        exit()
    except KeyboardInterrupt:
        exit()
    except CvBridgeError as e:
        print(e)


def main():
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

