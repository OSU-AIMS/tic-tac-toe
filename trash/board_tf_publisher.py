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

# Add game_scripts directory to the python-modules path options to allow importing other python scripts
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '//home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/game_scripts')
ttt_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_2_game_scripts = ttt_pkg + '/game_scripts'
sys.path.insert(1, path_2_game_scripts)

import rospy
import tf2_ros
import tf2_msgs.msg
import tf.transformations as tr

# ROS Data Types
from sensor_msgs.msg import Image
import geometry_msgs.msg
# from geometry_msgs import TransformStamped 

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



#####################################################
## SUPPORT CLASSES AND FUNCTIONS
##
def define_board_tile_centers():
    """
    Uses hard-coded scale values for size of TTT board to create a list of transformations for each board-tile center
    relative to the Board Origin (ie, center of board.. tile 4 center)
    tictactoe board order assignment:
    [0 1 2]
    [3 4 5]
    [6 7 8]

    :return: List of Transformation Matrices for each board
    """

    # Hard-Coded Board Scale Variables (spacing between tile centers)
    #centerxDist = 0.05863
    #centeryDist = -0.05863
    centerxDist = 0.0635
    centeryDist = 0.0635
    pieceHeight = 0.0

    # Build 3x3 Matrix to Represent the center of each TTT Board Tile.
    # board_tile_locations =[[-centerxDist ,centeryDist,pieceHeight],[0,centeryDist,pieceHeight],[centerxDist,centeryDist,pieceHeight],
    #                        [-centerxDist,0,pieceHeight],[0,0,pieceHeight],[centerxDist,0,pieceHeight],
    #                        [-centerxDist,-centeryDist,pieceHeight],[0,-centeryDist,pieceHeight],[centerxDist,-centeryDist,pieceHeight]]

    board_tile_locations =[[pieceHeight,-centerxDist ,centeryDist ],[pieceHeight, 0,centeryDist],[pieceHeight, centerxDist,centeryDist],
                           [pieceHeight,-centerxDist,0],[pieceHeight,0,0],[pieceHeight,centerxDist,0],
                           [pieceHeight,-centerxDist,-centeryDist],[pieceHeight,0,-centeryDist],[pieceHeight, centerxDist,-centeryDist,]]

    # Convert to Numpy Array
    tictactoe_center_list = np.array(board_tile_locations, dtype=np.float)

    # Set Default Rotation matrix (each tile rotation square with the board)
    rot_default = np.identity(3)
    tile_locations_tf = []

    # Create a list of TF's representing each Tile Center. Final list is 9-elements long
    tf_helper = transformations()
    for vector in tictactoe_center_list:
        item = np.matrix(vector)
        tile_locations_tf.append( tf_helper.generateTransMatrix(rot_default, item) )

    return tile_locations_tf

def msg_to_se3(msg):
    """Conversion from geometric ROS messages into SE(3)

    @param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
    C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
    @return: a 4x4 SE(3) matrix as a numpy array
    @note: Throws TypeError if we receive an incorrect type.
    """
    if isinstance(msg, geometry_msgs.msg.TransformStamped):
        p, q = transform_to_pq(msg.transform)
    else:
        raise TypeError("Invalid type for conversion to SE(3)")
    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-3:
        raise ValueError(
            "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                str(q), np.linalg.norm(q)))
    elif np.abs(norm - 1.0) > 1e-6:
        q = q / norm
    g = tr.quaternion_matrix(q)
    g[0:3, -1] = p
    return g

def transform_to_pq(msg):
    """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
    q = np.array([msg.rotation.x, msg.rotation.y,
                  msg.rotation.z, msg.rotation.w])
    return p, q

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
    # print(image.shape)

    # Find all contours in image
    #contours = shapeDetect.getContours(image)

    # Find the board contour, create visual, define board center and points
    boardImage, boardCenter, boardPoints = shapeDetect.detectSquare(image, area=23000)

    # Scale from image pixels to m (pixels/m)
    scale = .664/640          # res: (640x480)
    # scale = .895 / 1280         # res: (1280x730)
    # TODO: use camera intrinsics

    scaledCenter = [0, 0]

    # TODO check if works:
    # scaledCenter[0] = (boardCenter[0]-data.width / 2) * scale
    # scaledCenter[1] = (boardCenter[1]-data.height / 2) * scale

    # Convert board center pixel values to meters (and move origin to center of image)
    scaledCenter[0] = (boardCenter[0] - 320) * scale
    scaledCenter[1] = (boardCenter[1] - 240) * scale

    # Define 3x1 array of board translation (x, y, z) in meters
    boardTranslation = np.array(
        [[.655], [-scaledCenter[0]], [-scaledCenter[1]]])  ## TODO use depth from camera data
        # [[scaledCenter[0]], [scaledCenter[1]], [-0.655]])

    # Find rotation of board on the table plane, only a z-axis rotation angle
    z_orient = -shapeDetect.newOrientation(boardPoints)

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
    tf_camera2board = tf_helper.generateTransMatrix(boardRotationMatrix, boardTranslation)


    return scaledCenter, boardImage, tf_camera2board

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

#####################################################
## PRIMARY CLASS
##

class board_publisher():
    """
     Custom tictactoe publisher class that:
     1. publishes a topic with the board to world (robot_origin) transformation matrix
     2. publishes a topic with the board tile center location on the image (pixel values)
     3. publishes a topic with an image that has board visuals
     4. creates a live feed that visualizes where the camera thinks the board is located
    """

    def __init__(self, camera2board_pub, center_pub, camera_tile_annotation, tfBuffer):

        # Inputs
        self.camera2board_pub = camera2board_pub

        self.center_pub = center_pub
        # center_pub: publishes the board to world transformation matrix

        self.camera_tile_annotation = camera_tile_annotation
        # camera_tile_annotation: publishes the numbers & arrows displayed on the image

        self.tfBuffer = tfBuffer
        # tfBuffer: listener for all all transforms in ROS

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

            # characterize board location and orientation

            # Run using color (use HSV extraction to obtain accurate HSV values)
            # scaledCenter, boardImage, tf_camera2board = detectBoard_coloredSquares(cv_image)
            
            # Run using contours
            scaledCenter, boardImage, tf_camera2board = detectBoard_contours(cv_image)


            
            # Distance from Camera to Board
            # .. Hacky. Extract height from camera global transform.
            # standoff_z_axis = tf_matrix.transform.translation.z


            pose_goal = tf_helper.transformToPose(tf_camera2board)

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
            # rospy.loginfo(camera2board_msg)

            fixed2board_tf = self.tfBuffer.lookup_transform('base_link', 'ttt_board', rospy.Time(0))
            self.center_pub.publish(fixed2board_tf)
            rospy.loginfo(fixed2board_tf)

            fixed2board_matrix = msg_to_se3(fixed2board_tf)
            
            tileCentersMatrices = define_board_tile_centers()
            tileCenters2camera = tf_helper.convertPath2FixedFrame(tileCentersMatrices, fixed2board_matrix)  # 4x4 transformation matrix
            
            xyList = [[] for i in range(9)]
            # scale = .895 / 1280  # todo: set to camera intrinsics
            scale = .664 / 640
            for i in range(9):
                xyzCm = (tileCenters2camera[i][0:2, 3:4])  # in cm
                x = xyzCm[0] / scale + 320
                y = xyzCm[1] / scale + 240  # in pixels
                xyList[i].append(int(x))
                xyList[i].append(int(y))
                cv2.putText(boardImage, str(i), (int(xyList[i][0]), int(xyList[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 0),
                            2)

            # save pixel locations for tiles
            tictactoe_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            filename = 'tile_centers_pixel.npy'

            outputFilePath = tictactoe_pkg + '/' + filename
            np.save(outputFilePath, xyList)

            # Image Stats
            height, width, channels = boardImage.shape

            # Prepare Image Message
            # msg_img = Image()
            # msg_img.height = height
            # msg_img.width = width
            # msg_img.encoding = 'rgb8'
            try:
                msg_img = self.bridge.cv2_to_imgmsg(boardImage, 'bgr8')
            except CvBridgeError as e:
                print(e)

            # Publish
            self.camera_tile_annotation.publish(msg_img)
            #rospy.loginfo(msg)

            # cv2.imshow('CV2: Live Board', boardImage)
            # cv2.waitKey(3)


        except rospy.ROSInterruptException:
            exit()
        except KeyboardInterrupt:
            exit()
        except CvBridgeError as e:
            print(e)


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
    pub_center = rospy.Publisher("ttt_board_origin",geometry_msgs.msg.TransformStamped,queue_size=20)
    pub_camera_tile_annotation = rospy.Publisher("camera_tile_annotation", Image, queue_size=20)

    # rate = rospy.Rate(0.1)


    # Setup Listeners
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    bp_callback = board_publisher(pub_camera2board, pub_center, pub_camera_tile_annotation, tfBuffer)
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, bp_callback.runner)


    # Auto-Run until launch file is shutdown
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
