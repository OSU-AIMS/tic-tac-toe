#!/usr/bin/env python

#####################################################
#   Support Node to Output Board Position           #
#                                                   #
#   * Works Primarily in transforms                 #
#   * Relies upon camera input topic                #
#   * Publishes multiple output topics for results  #
#                                                   #
#####################################################
# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#####################################################



#####################################################
## IMPORTS
import sys
import os
from os.path import join, abspath, dirname

ttt_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_2_scripts = ttt_pkg + '/scripts'
sys.path.insert(1, path_2_scripts)

import rospy
import tf2_ros
import tf2_msgs.msg

# ROS Data Types
from sensor_msgs.msg import Image
import geometry_msgs.msg
from std_msgs.msg import ByteMultiArray

# Custom Tools
  # from Realsense_tools import *
from toolbox_shape_detector import *
from cv_bridge import CvBridge, CvBridgeError

# System Tools
from math import pi, radians, sqrt, atan
import numpy as np


# Ref
# http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20listener%20%28Python%29
# http://docs.ros.org/en/jade/api/geometry_msgs/html/msg/Transform.html
shapeDetect = TOOLBOX_SHAPE_DETECTOR()



def detectBoard_contours(image):
    """
    Tictactoe function that finds the physical board. Utilizes ShapeDetector class functions.
    TODO transition to using rgb dots for both board center and orientation.
    @param image: image parameter the function tries to find the board on.
    @return scaledCenter: (x ,y) values in meters of the board center relative to the center of the camera frame.
    @return boardImage: image with drawn orientation axes and board location.
    @return tf_camera2board: transformation matrix of board to camera frame of reference.
    """

    # Reading in static image
    # frame = cv2.imread('../sample_content/sample_images/1X_1O_ATTACHED_coloredSquares_Color_Color.png')

    image = image.copy()

    # Find the board contour, create visual, define board center and points
    boardImage, boardCenter, boardPoints = shapeDetect.detectSquare(image, area=54600)

    # Scale from image pixels to m (pixels/m)
    # scale = .664/640          # res: (640x480)
    scale = 1.14 / 1280         # res: (1280x730)
    # TODO: use camera intrinsics

    scaledCenter = [0, 0]

    # TODO check if works:
    # scaledCenter[0] = (boardCenter[0]-data.width / 2) * scale
    # scaledCenter[1] = (boardCenter[1]-data.height / 2) * scale

    # Convert board center pixel values to meters (and move origin to center of image)
    scaledCenter[0] = (boardCenter[0] - 640) * scale
    scaledCenter[1] = (boardCenter[1] - 360) * scale

    # Define 3x1 array of board translation (x, y, z) in meters
    boardTranslation = np.array(
        [[.819], [scaledCenter[0]], [-scaledCenter[1]]])  ## TODO use depth from camera data
        # [[scaledCenter[0]], [scaledCenter[1]], [-0.655]])

    # Find rotation of board on the table plane, only a z-axis rotation angle
    z_orient = -shapeDetect.findAngle(boardPoints)

    # convert angle to a rotation matrix with rotation about z-axis
    board_rot = np.array([[math.cos(radians(z_orient)), -math.sin(radians(z_orient)), 0],
                        [math.sin(radians(z_orient)), math.cos(radians(z_orient)), 0],
                        [0, 0, 1]])

    y_neg90 = np.array([[ 0,  0, -1],
                        [0,  1,  0],
                        [1,  0,  0]])

    z_neg90 = np.array([[0,1,0],
                        [-1,0,0],
                        [0,0,1]])

    camera_rot = np.dot(y_neg90,z_neg90)

    # board_rot = np.array([[1, 0, 0],
    #                                 [0,math.cos(radians(z_orient)), -math.sin(radians(z_orient))],
    #                                 [0,math.sin(radians(z_orient)), math.cos(radians(z_orient))]])

    boardRotationMatrix = np.dot(camera_rot,board_rot)

    # Transformation (board from imageFrame to camera)

    # Build new tf matrix

    tf_camera2board = np.zeros((4, 4))
    tf_camera2board[0:3, 0:3] = boardRotationMatrix
    tf_camera2board[0:3, 3:4] = boardTranslation
    tf_camera2board[3, 3] = 1

    return scaledCenter, boardImage, tf_camera2board

def detectColored_Markers(image):
    # colored_marker_centers = rospy.wait_for_message('Fiducial_Centers',ByteMultiArray,timeout = 10)
    # print(colored_marker_centers)

    readings = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # Create kernel (format - "BGR")
    # kernel_size = 5
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
    # print('min_loc_G', min_loc_G)
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
    # print('endX_G',endX_G)
    
    # # show the output image - uncomment to show all 3 squares detected
    # if you want to see just red and blue, then use cv2.imshow('Red Detect',r_box_image) beneath the red square detection
    cv2.imshow("Green Detect", g_box_image)

    # cv2.imwrite('res_match_template_GREEN_BoundingBox.tiff', g_box_image)
    cv2.waitKey(1)

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

    x_blue = (startX_B+endX_B)/2
    y_blue = (startY_B+endY_B)/2
    center_blue = [x_blue,y_blue]

    x_red = (startX_R+endX_R)/2
    y_red = (startY_R+endY_R)/2
    center_red = [x_red,y_red]

    x_green = (startX_G+endX_G)/2
    y_green = (startY_G+endY_G)/2
    center_green = [x_green,y_green]


    # print('Center Blue', center_blue)
    # print('Center Red', center_red)
    # print('Center Green',center_green)

    return center_blue, center_green, center_red

def transformToPose(transform):
    # Location Vector
    pose_goal = []
    point = transform
    x, y, z = point[:-1, 3]
    x = np.asscalar(x)
    y = np.asscalar(y)
    z = np.asscalar(z)

    # Quant Calculation Support Variables
    # Only find trace for the rotational matrix.
    t = np.trace(point) - point[3, 3]
    r = np.sqrt(1 + t)

    # Primary Diagonal Elements
    Qxx = point[0, 0]
    Qyy = point[1, 1]
    Qzz = point[2, 2]

    # Quant Calculation
    qx = np.copysign(0.5 * np.sqrt(1 + Qxx - Qyy - Qzz), point[2, 1] - point[1, 2])
    qy = np.copysign(0.5 * np.sqrt(1 - Qxx + Qyy - Qzz), point[0, 2] - point[2, 0])
    qz = np.copysign(0.5 * np.sqrt(1 - Qxx - Qyy + Qzz), point[1, 0] - point[0, 1])
    qw = 0.5 * r

    pose_goal = [x, y, z, qx, qy, qz, qw]
    return pose_goal

def calcFiducialCentroid(blue_center, green_center, red_center, rotMatrix):
    """
    Find Maze Centroid in Image using Centroids of the 3 Dots
    :param dots: Input Dictionary with three dot keys.
    :return rot_matrix: 3x3 Rotation Matrix.
    """
    
    # Convention:   Right-Hand-Rule
    #   Origin  :   @ BLUE
    #   X-Vector:   BLUE -> RED
    #   Y-VECTOR:   BLUE -> GREEN

    # Use Numpy for Calculation Convenience

    # Find Vectors
    # side_br = np.subtract(red_center,   blue_center)
    x_midpoint = (red_center[0]+green_center[0])/2
    y_midpoint = (red_center[1]+green_center[1])/2
    mazeCentroid = np.array([x_midpoint, y_midpoint])  

    side_br = np.subtract(blue_center,red_center)
    side_bg = np.subtract(blue_center,green_center)

    len_br = np.linalg.norm(side_br)
    len_bg = np.linalg.norm(side_bg)

    #MAZE SIZE: .203m x .203m
    maze_size_meters = [.203,.203]
    scale_br = maze_size_meters[0] / len_br
    scale_bg = maze_size_meters[1] / len_bg
    scale = np.average([scale_br,scale_bg]) 
    
    return mazeCentroid, scale

def calcRotation(blue_center, green_center, red_center):

    # use point-slope formula
    BR_slope = (float(blue_center[1]) - float(red_center[1])) / (float(blue_center[0]+1E-7) - float(red_center[0]))
    # BG_slope = (float(blue_center[1]) - float(green_center[1])) / (float(blue_center[0]) - float(green_center[0]))


    # convert to radians
    angle = -1*math.degrees(math.atan(BR_slope))

    rot_matrix = np.array([[math.cos(radians(angle)), -math.sin(radians(angle)), 0],
                        [math.sin(radians(angle)), math.cos(radians(angle)), 0],
                        [0, 0, 1]])
    
    return rot_matrix


def mean(nums):
    return float(sum(nums)) / max(len(nums),1)

class board_publisher():
    """
     Custom tictactoe publisher class that:
     1. publishes a topic with the board to world (robot_origin) transformation matrix
     2. publishes a topic with the board tile center location on the image (pixel values)
     3. publishes a topic with an image that has board visuals
     4. creates a live feed that visualizes where the camera thinks the board is located
    """

    def __init__(self, camera2board_pub):

        # Inputs
        self.camera2board_pub = camera2board_pub

        # Tools
        self.bridge = CvBridge()



    def runner(self, data):
        """
        Callback function for image subscriber, every frame gets scanned for board and publishes to board_center topic
        (for robot movement) and board tile centers (for game state updates)
        :param camera_data: Camera data input from subscriber
        """
        try:
            # ToDo: Check if this works. Or default back to [640,360] (only highlighted in PyCharm)
            boardCenter = [data.width/2, data.height/2]   # Initialize as center of frame

            # Convert Image to CV2 Frame
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            boardImage = cv_image.copy()
           
            # Run using contours
            # scaledCenter, boardImage, tf_camera2board = self.detectBoard_contours(cv_image)
            center_blue, center_green, center_red = detectColored_Markers(cv_image)

            tf_camera2board = self.matrixColored_Markers(center_blue, center_green, center_red)

            pose_goal = transformToPose(tf_camera2board)

            ## Publish Board Pose
            camera2board_msg = geometry_msgs.msg.TransformStamped()
            camera2board_msg.header.frame_id = 'camera_link'
            camera2board_msg.child_frame_id = 'ttt_board'
            camera2board_msg.header.stamp = rospy.Time.now()

            camera2board_msg.transform.translation.x = pose_goal[0]
            camera2board_msg.transform.translation.y = pose_goal[1]
            camera2board_msg.transform.translation.z = pose_goal[2]

            camera2board_msg.transform.rotation.x = pose_goal[3]
            camera2board_msg.transform.rotation.y = pose_goal[4]
            camera2board_msg.transform.rotation.z = pose_goal[5]
            camera2board_msg.transform.rotation.w = pose_goal[6]

            camera2board_msg = tf2_msgs.msg.TFMessage([camera2board_msg])

            # Publish
            self.camera2board_pub.publish(camera2board_msg)            

        except rospy.ROSInterruptException:
            exit()
        except KeyboardInterrupt:
            exit()
        except CvBridgeError as e:
            print(e)


    def matrixColored_Markers(self,b,g,r):
        board_rot = calcRotation(b,g,r)
        board_center, scale = calcFiducialCentroid(b,g,r,board_rot)

        print('board_center',board_center)

        # self.board_center_readingX = np.array([0,0,0,0,0,0,0,0,0])
        # self.board_center_readingY = np.array([0,0,0,0,0,0,0,0,0])
        # MAX_SAMPLES = 10

        # self.board_center_readingX = np.append(self.board_center_readingX, board_center[0])
        # board_center_avgX = mean(self.board_center_readingX)

        # self.board_center_readingY = np.append(self.board_center_readingY, board_center[1])
        # board_center_avgY = mean(self.board_center_readingY)

        # print('Board center averageX:',board_center_avgX)
        # print('Board center averageY:',board_center_avgY)


        # if len(self.board_center_readingX) == MAX_SAMPLES:
        #     self.board_center_readingX = np.delete(self.board_center_readingX,0)
        #     self.board_center_readingY = np.delete(self.board_center_readingY,0)

        board_center_avgX = board_center[0]
        board_center_avgY = board_center[1]

        # print('Board center averageX:',board_center_avgX)
        # print('Board center averageY:',board_center_avgY)

        scaledCenter = [0, 0]

        scale = 1.14/1280

        scaledCenter[0] = (board_center_avgX - 640) * scale
        scaledCenter[1] = (board_center_avgY - 360) * scale

        # Define 3x1 array of board translation (x, y, z) in meters
        boardTranslation = np.array(
            [[.819], [scaledCenter[0]], [-scaledCenter[1]]])  

        y_neg90 = np.array([[ 0,  0, -1],
                            [0,  1,  0],
                            [1,  0,  0]])

        z_neg90 = np.array([[0,1,0],
                            [-1,0,0],
                            [0,0,1]])

        camera_rot = np.dot(y_neg90,z_neg90)

        boardRotationMatrix = np.dot(camera_rot,board_rot)

        # Build new tf matrix

        tf_camera2board = np.zeros((4, 4))
        tf_camera2board[0:3, 0:3] = boardRotationMatrix
        tf_camera2board[0:3, 3:4] = boardTranslation
        tf_camera2board[3, 3] = 1

        return tf_camera2board





#####################################################
## MAIN()
def main():
    """
    Main Runner.
    This script should only be launched via a launch script or when the Camera Node is already open.
        ttt_board_origin: publishes the board center and rotation matrix
        camera_tile_annotation: publishes the numbers & arrows displayed on the image

    """

    # Setup Node
    rospy.init_node('board_vision_processor', anonymous=False)
    rospy.loginfo(">> Board Vision Processor Node Successfully Created")


    # Setup Publishers
    pub_camera2board = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=20)

    # Setup Listeners
    bp_callback = board_publisher(pub_camera2board)
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, bp_callback.runner)


    # Auto-Run until launch file is shutdown
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()