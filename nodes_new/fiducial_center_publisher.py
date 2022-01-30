#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2022, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: MohammadKhan-3


#####################################################

# Imports
import cv2
import rospy
from os.path import join, abspath, dirname

# ROS Data Types
from sensor_msgs.msg import Image
from std_msgs.msg import ByteMultiArray
from std_msgs.msg import MultiArrayDimension
import tf2_ros


# System Tools
from math import pi, radians, sqrt, atan, ceil
import numpy as np
import matplotlib.pyplot as plt

# Custom Tool
from cv_bridge import CvBridge, CvBridgeError


# RealSense Pipeline
import pyrealsense2 as rs 
import time
import sys


def kernel_runner(image):
    # Create kernel (format - "BGR")
    kernel_size = 5
    # print('Image Parameter')
    # print(image)
    # print('Inside Kernel_Runner Function')
    # print('Shape of Input image')
    # print(np.shape(image))
    # print('Type for Input Image')
    # print(type(image))


    # Uncomment below to use square images as kernels
    CWD = dirname(abspath(__file__)) # Control Working Directory - goes to script location
    RESOURCES = join(CWD,'image_kernels') # combine script location with folder name
    # blue_square = 'blue_square_crop.tiff'  - Used for Static Image and other images at the same depth & focal Length
    blue_square = 'blue-circle-square.tiff'
    red_square = 'red-circle-square.tiff'
    green_square = 'green-circle-square.tiff'

    kernel_b = cv2.imread(join(RESOURCES,blue_square)) # combine folder name with picture name inside folder
    
    # Uncomment to check if the file path to the image kernel is correct
    # print('Join Resources')
    # print(join(RESOURCES,blue_square))

    # print('Type for Kernel_b Template img')
    # print(type(kernel_b))

    kernel_g = cv2.imread(join(RESOURCES,green_square))

    kernel_r = cv2.imread(join(RESOURCES,red_square))

    # Uncomment below to see Kernel size
    # print('Kernel Matrix: should be 3x3x3')
    # print(np.shape(kernel_b)) # returns 3x3x3
    # print(kernel_b)

    '''
    matchTemplate docs
    https://docs.opencv.org/4.2.0/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dac6677e2af5e0fae82cc5339bfaef5038
        Input Array: image (must be 8 bit or 32 bit floating point)
        Input array: Templ (serached template)
        output array: result
               int: method (https://docs.opencv.org/4.2.0/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d)
               mask: mask of serached template. Same datatype & size as templ. Not set by default
    '''

    # # Recognizing Blue Square --- Everything needed to run matchTemplate below
    # print('Using matchTemplate() function')
    image = image.copy()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res_B = cv2.matchTemplate(image=image,templ=kernel_b,method=5)
    # res_B = cv2.matchTemplate(image=img_gray,templ=kernel_b_gray,method=5)
    # Use method=5 when using the square images as kernels
    cv2.imwrite('res_match_template_B.tiff', res_B)
    min_val_B, max_val_B, min_loc_B, max_loc_B = cv2.minMaxLoc(res_B)
    # print('min_val_B')
    # print(min_val_B)
    # print('max_val_B')
    # print(max_val_B)
    # print(' ')
    # print('min_loc_B')
    # print(min_loc_B)
    # print('max_loc_B')
    # print(max_loc_B)

    # Drawing Bounding Box around detected shape
    # determine the starting and ending (x, y)-coordinates of the bounding box
    # From: https://www.pyimagesearch.com/2021/03/22/opencv-template-matching-cv2-matchtemplate/
    (startX_B, startY_B) = max_loc_B
    endX_B = startX_B + kernel_b.shape[1]
    endY_B = startY_B + kernel_b.shape[0]

    # draw the bounding box on the image
    b_box_image = cv2.rectangle(image, (startX_B, startY_B), (endX_B, endY_B), (255, 0, 0), 4) # BGR for openCV

 #------------ Everything needed for matchTemplate() ^^^

    #### Recognizing Red Square
    res_R = cv2.matchTemplate(image=image,templ= kernel_r,method=5)
    # cv2.imwrite('res_match_template_R.tiff', res_R)
    min_val_R, max_val_R, min_loc_R, max_loc_R = cv2.minMaxLoc(res_R)
    # # print('min_val_R')
    # # print(min_val_R)
    # # print('max_val_R')
    # # print(max_val_R)
    # print(' ')
    # print('min_loc_R')
    # print(min_loc_R)
    # print('max_loc_R')
    # print(max_loc_R)
    
    # # Drawing Bounding Box around detected shape
    # # determine the starting and ending (x, y)-coordinates of the bounding box
    # # From: https://www.pyimagesearch.com/2021/03/22/opencv-template-matching-cv2-matchtemplate/
    (startX_R, startY_R) = max_loc_R
    endX_R = startX_R + kernel_r.shape[1]
    endY_R = startY_R + kernel_r.shape[0]
    
    # # draw the bounding box on the image
    r_box_image = cv2.rectangle(image, (startX_R, startY_R), (endX_R, endY_R), (0, 0, 255), 3)

    #### Recognizing Green Square
    res_G = cv2.matchTemplate(image=image,templ= kernel_g,method=5)
    # cv2.imwrite('res_match_template_G.tiff', res_G)
    min_val_G, max_val_G, min_loc_G, max_loc_G = cv2.minMaxLoc(res_G)
    # # print('min_val_G')
    # # print(min_val_G)
    # # print('max_val_G')
    # # print(max_val_G)
    # print(' ')
    # print('min_loc_G')
    # print(min_loc_G)
    # print('max_loc_G')
    # print(max_loc_G)
    #
    # # Drawing Bounding Box around detected shape
    # # determine the starting and ending (x, y)-coordinates of the bounding box
    # # From: https://www.pyimagesearch.com/2021/03/22/opencv-template-matching-cv2-matchtemplate/
    (startX_G, startY_G) = max_loc_G
    endX_G = startX_G + kernel_g.shape[1]
    endY_G = startY_G + kernel_g.shape[0]
    #
    # # draw the bounding box on the image
    g_box_image = cv2.rectangle(image, (startX_G, startY_G), (endX_G, endY_G), (0, 255, 0), 3)
    
    # # show the output image - uncomment to show all 3 squares detected
    # if you want to see just red and blue, then use cv2.imshow('Red Detect',r_box_image) beneath the red square detection
    # cv2.imshow("Green Detect", g_box_image)

    # cv2.imwrite('res_match_template_GREEN_BoundingBox.tiff', g_box_image)
    # cv2.waitKey(1)

    '''
        cv::TemplateMatchModes 
        cv::TM_SQDIFF = 0,
        cv::TM_SQDIFF_NORMED = 1,
        cv::TM_CCORR = 2,
        cv::TM_CCORR_NORMED = 3,
        cv::TM_CCOEFF = 4,
        cv::TM_CCOEFF_NORMED = 5
    '''
    #### Using Bounding-Box Coordinates to get orientation of the board ## Output Bounding Box centers to different function
    # note this likely will be done in a separate function but test it here
    # center_B = (np.subtract(max_loc_B[0], min_loc_B[0]),np.subtract(max_loc_B[1],min_loc_B[1]))
    # center_G = np.subtract(max_loc_G, min_loc_G)
    # center_R = np.subtract(max_loc_R, min_loc_R)

    # print('Center_B Square:')
    # print(center_B)
    # shapeDetect.drawAxis()

    center_blue = [max_loc_B]
    center_green = [max_loc_G]
    center_red = [max_loc_R]
    
    return center_blue, center_green, center_red
    # Centers of the blue, green, red squares --> to be outputted 



class board_location_publisher():
    """
    Using image kernel to detect location of the blue, red, green circles of TTT board
    """
    def __init__(self, board_location):
        self.bridge = CvBridge()
        self.board_location =  board_location
 
    def runner(self, data):
        """
        Callback function for image subscriber, every frame gets scanned for board and publishes to board_center topic
        (for robot movement) and board tile centers (for game state updates)
        :param camera_data: Camera data input from subscriber
        """
        try:
            board_location = [0,0,0,0,0,0]
            # print("Inside Runner")
            # Convert Image to CV2 Frame
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
            # OpenCV:BGR / RealSense: RGB / RGB: to get proper colors --> also filps colors in frame

            # Type for cv_image debugging
            # print('Type for cv_image:')
            # print(type(cv_image))

            # Using Image Kernel to detect color
            blue,green,red = kernel_runner(cv_image)
            board_location = ByteMultiArray()
            data = (blue[0][0], blue[0][1], green[0][0],green[0][1], red[0][0],red[0][1])
            board_location.data = list(bytearray(data))
            board_location.layout.dim.append(MultiArrayDimension())
            board_location.dim[0].label = 'colored_centers'
            board_location.dim[0].size = len(board_location.data)
            board_location.dim[0].stride = len(board_location.data)
            board_location.data_offset = 0


            # Publish location of highest value in BGR
            self.board_location.publish(board_location)
            rospy.loginfo(board_location)


        except rospy.ROSInterruptException:
            exit()
        except KeyboardInterrupt:
            exit()
        except CvBridgeError as e:
            print(e)

def main():
    print("Your OpenCV version is: " + cv2.__version__)  

    # Initialize a Node:
    rospy.init_node('Board_Location', anonymous=False)
    rospy.loginfo(">> Board Location Node Successfully Created")

    # Setup Publishers
    pub_board_location = rospy.Publisher("Fiducial_Centers", ByteMultiArray, queue_size=20)


    bl_callback = board_location_publisher(pub_board_location)

    # create subscriber to ros Image Topic - pulled from kernel_color_detect
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, bl_callback.runner)


    # print('Type of Image from image_sub')
    # print(type(image_sub)) # type 'numpy.ndarray'

    print("Subscribed to image_raw!")   

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()

