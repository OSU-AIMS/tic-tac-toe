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
    centeryDist = -0.0635
    pieceHeight = -0.0

    # Build 3x3 Matrix to Represent the center of each TTT Board Tile.
    board_tile_locations =[[-centerxDist ,centeryDist,pieceHeight],[0,centeryDist,pieceHeight],[centerxDist,centeryDist,pieceHeight],
                           [-centerxDist,0,pieceHeight],[0,0,pieceHeight],[centerxDist,0,pieceHeight],
                           [-centerxDist,-centeryDist,pieceHeight],[0,-centeryDist,pieceHeight],[centerxDist,-centeryDist,pieceHeight]]

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


def detectBoard(image):
    """
    Tictactoe function that finds the physical board. Utilizes ShapeDetector class functions.
    TODO transition to using rgb dots for both board center and orientation.
    @param image: image parameter the function tries to find the board on.
    @return scaledCenter: (x ,y) values in meters of the board center relative to the center of the camera frame.
    @return boardImage: image with drawn orientation axes and board location.
    @return tf_board2camera: transformation matrix of board to camera frame of reference.
    """

    # Reading in static image
    # frame = cv2.imread('../sample_content/sample_images/1X_1O_ATTACHED_coloredSquares_Color_Color.png')

    image = image.copy()

    # Find all contours in image
    #contours = shapeDetect.getContours(image)

    # Find the board contour, create visual, define board center and points
    boardImage, boardCenter, boardPoints = shapeDetect.detectSquare(image, area=90000)

    # Scale from image pixels to m (pixels/m)
    # scale = .664/640          # res: (640x480)
    scale = .895 / 1280         # res: (1280x730)

    scaledCenter = [0, 0]

    # TODO check if works:
    # scaledCenter[0] = (boardCenter[0]-data.width / 2) * scale
    # scaledCenter[1] = (boardCenter[1]-data.height / 2) * scale

    # Convert board center pixel values to meters (and move origin to center of image)
    scaledCenter[0] = (boardCenter[0] - 640) * scale
    scaledCenter[1] = (boardCenter[1] - 360) * scale

    # Define 3x1 array of board translation (x, y, z) in meters
    boardTranslation = np.array(
        [[scaledCenter[0]], [scaledCenter[1]], [0.655]])  ## depth of the table is .655 m from camera

    # Find rotation of board on the table plane, only a z-axis rotation angle
    z_orient = shapeDetect.newOrientation(boardPoints)

    # shapeDetect.drawAxis(img, cntr, p1, (255, 255, 0), 1)
    # shapeDetect.drawAxis(img, cntr, p2, (255, 80, 255), 1)

    # convert angle to a rotation matrix with rotation about z-axis
    boardRotationMatrix = np.array([[math.cos(radians(z_orient)), -math.sin(radians(z_orient)), 0],
                                    [math.sin(radians(z_orient)), math.cos(radians(z_orient)), 0],
                                    [0, 0, 1]])

    # Transformation (board from imageFrame to camera)
    tf_board2camera = tf_helper.generateTransMatrix(boardRotationMatrix, boardTranslation)

    return scaledCenter, boardImage, tf_board2camera

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
    scale = .895 / 1280         # res: (1280x730)

    # cv2.imshow("Image Frame",imageFrame)
    # cv2.waitKey(3)
    # used for debugging purposes

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    cv2.imshow('HSV frame',hsvFrame)
    cv2.waitKey(3)

    # Set range for red color and define mask
    # red_lower = np.array([136, 87, 111], np.uint8)
    # red_upper = np.array([180, 255, 255], np.uint8)

    # are thresholds below rbg (not this) or rgb(not this) or bgr(not this) or gbr(not this)
    # psych: It's converting BGR to HSV

    # Set range for red color and define mask
    red_lower = np.array([0, 190, 210], np.uint8)
    # [0, 190, 220]
    red_upper = np.array([2, 220, 255], np.uint8)
    # [2, 220, 255]
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for green color and define mask
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
    scaledCenter[0] = ((x_axis[0]+x_axis[1])/2 - 640) * scale
    scaledCenter[1] = ((y_axis[0]+y_axis[1])/2 - 365) * scale #originally subtracted 360, changed to 365 = 720/2
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
    tf_board2camera = tf_helper.generateTransMatrix(boardRotationMatrix, boardTranslation)

    # do I need to return BoardImage
    return scaledCenter, tf_board2camera 


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

    def __init__(self, center_pub, camera_tile_annotation):

        # Inputs

        self.center_pub = center_pub
        # center_pub: publishes the board to world transformation matrix

        self.camera_tile_annotation = camera_tile_annotation
        # camera_tile_annotation: publishes the numbers & arrows displayed on the image


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

            # Run using color
            scaledCenter, tf_board2camera = detectBoard_coloredSquares(cv_image)
            
            # Run using contours
            # scaledCenter, boardImage, tf_board2camera = detectBoard(cv_image)
        
            

            # For visual purposes, simple crop
            # cropped_image = frame[center[1]-90:center[1]+90,center[0]-90:center[0]+90] #640x480
            # cropped_image = frame[center[1] - 125:center[1] + 125, center[0] - 125:center[0] + 125]  # 1280x720

            # find all 9 nine tile centers based on board center
            tileCentersMatrices = define_board_tile_centers()

            tileCenters2camera = tf_helper.convertPath2FixedFrame(tileCentersMatrices, tf_board2camera)  # 4x4 transformation matrix

            # Columns: 0,1,2 are rotations, column: 3 is translation
            # Rows: 0,1 are x & y rotation & translation values


            import subprocess

            ## Build Camera_2_World TF
            # todo: Rewrite how the camera-tf is defined. Suggestion: grab from topic 'tf'
            # ttt_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            # tf_camera2world_filepath = np.load(ttt_pkg + "/tf_camera2world.npy")
            tfBuffer = tf2_ros.Buffer()
            listener = tf2_ros.TransformListener(tfBuffer)

            data = tfBuffer.lookup_transform('base_link', 'camera_link', rospy.Time())
            tf_camera2world_vector = [ 
                data.transform.translation.x,
                data.transform.translation.y,
                data.transform.translation.z, ### change this value to hardcoded z-value
                data.transform.rotation.x,
                data.transform.rotation.y,
                data.transform.rotation.z,
                data.transform.rotation.w
                ]
            #TODO: Use native techniques. http://docs.ros.org/en/melodic/api/tf_conversions/html/python/
            tf_camera2world = tf_helper.quant_pose_to_tf_matrix(tf_camera2world_vector)


            rot_camera_hardcode = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])

            # translation = tf_camera2world[:-1, -1].tolist()
            # tf_camera2world = tf_helper.generateTransMatrix(rot_camera_hardcode, translation)

            ## Build Board_2_World TF
            tf_board2world = np.matmul(tf_camera2world, tf_board2camera)

            ## Convert TF (4x4) Array to Pose (7-element list)
            pose_goal = tf_helper.transformToPose(tf_board2world)

            ## Publish Board Pose
            msg = TransformStamped()
            msg.header.frame_id = 'Origin'# change to camera link frame id (then just lookup transform for board to origin)
            msg.child_frame_id = 'Board'
            msg.transform.translation.x = pose_goal[0]
            msg.transform.translation.y = pose_goal[1]
            msg.transform.translation.z = pose_goal[2]
            msg.transform.rotation.x = pose_goal[3]
            msg.transform.rotation.y = pose_goal[4]
            msg.transform.rotation.z = pose_goal[5]
            msg.transform.rotation.w = pose_goal[6]


            # Publish
            self.center_pub.publish(msg)
            rospy.loginfo(msg)


            # Draw Tile Numbers onto Frame
            xyList = [[] for i in range(9)]
            scale = .895 / 1280  # todo: set to camera intrinsics
            for i in range(9):
                xyzCm = (tileCenters2camera[i][0:2, 3:4])  # in cm
                x = xyzCm[0] / scale + 640
                y = xyzCm[1] / scale + 360  # in pixels
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
    pub_center = rospy.Publisher("ttt_board_origin", TransformStamped, queue_size=20)
    pub_camera_tile_annotation = rospy.Publisher("camera_tile_annotation", Image, queue_size=20)

    # rospy.Rate(0.1)


    # Setup Listeners
    bp_callback = board_publisher(pub_center, pub_camera_tile_annotation)
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, bp_callback.runner)


    # Auto-Run until launch file is shutdown
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

