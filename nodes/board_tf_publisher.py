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

ttt_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_2_game_scripts = ttt_pkg + '/game_scripts'
sys.path.insert(1, path_2_game_scripts)

import rospy

# ROS Data Types
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

# Custom Tools
from transformations import *
from shape_detector import *
from cv_bridge import CvBridge, CvBridgeError

# System Tools
import cv2
import time
import math
from math import pi, radians, sqrt
import numpy as np

# Ref
# http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20listener%20%28Python%29
# http://docs.ros.org/en/jade/api/geometry_msgs/html/msg/Transform.html

tf = transformations()
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

    ## Hard-Coded Board Scale Variables (spacing between tile centers in meters)

    # paper board:
    # centerxDist = 0.05863
    # centeryDist = -0.05863

    #3d printed board:
    centerxDist = 0.0635
    centeryDist = -0.0635
    pieceHeight = -0.0

    # Build 3x3 Matrix to Represent the center of each TTT Board Tile.
    board_tile_locations = [[-centerxDist, centeryDist, pieceHeight], [0, centeryDist, pieceHeight],
                            [centerxDist, centeryDist, pieceHeight],
                            [-centerxDist, 0, pieceHeight], [0, 0, pieceHeight], [centerxDist, 0, pieceHeight],
                            [-centerxDist, -centeryDist, pieceHeight], [0, -centeryDist, pieceHeight],
                            [centerxDist, -centeryDist, pieceHeight]]

    # Convert to Numpy Array
    tictactoe_center_list = np.array(board_tile_locations, dtype=np.float)

    # Set Default Rotation matrix (each tile rotation square with the board)
    rot_default = np.identity(3)
    tile_locations_tf = []

    # Create a list of TF's representing each Tile Center. Final list is 9-elements long
    for vector in tictactoe_center_list:
        item = np.matrix(vector)
        tile_locations_tf.append(tf.generateTransMatrix(rot_default, item))

    return tile_locations_tf


#####################################################
## PRIMARY CLASS
##

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

    shapeDetect.drawAxis(img, cntr, p1, (255, 255, 0), 1)
    shapeDetect.drawAxis(img, cntr, p2, (255, 80, 255), 1)

    # convert angle to a rotation matrix with rotation about z-axis
    boardRotationMatrix = np.array([[math.cos(radians(z_orient)), -math.sin(radians(z_orient)), 0],
                                    [math.sin(radians(z_orient)), math.cos(radians(z_orient)), 0],
                                    [0, 0, 1]])

    # Transformation (board from imageFrame to camera)
    tf_board2camera = tf.generateTransMatrix(boardRotationMatrix, boardTranslation)

    return scaledCenter, boardImage, tf_board2camera


class board_publisher:
    """
    Custom tictactoe publisher class that:
     1. publishes a topic with the board to world (robot_origin) transformation matrix
     2. publishes a topic with the board tile center location on the image (pixel values)
     3. publishes a topic with an image that has board visuals
     4. creates a live feed that visualizes where the camera thinks the board is located
    """
    def __init__(self):

        # Setup Publishers
        self.center_transform_pub = rospy.Publisher("ttt_board_origin", TransformStamped, queue_size=20)
        # ttt_board_origin: publishes the board to world transformation matrix

        self.camera_tile_annotation = rospy.Publisher("camera_tile_annotation", Image, queue_size=20)
        # camera_tile_annotation: publishes the numbers & arrows displayed on the image

        # self.tile_center_transforms = rospy.Publisher("tile_center_transforms", TransformStamped, queue_size=20)
        rospy.Rate(0.1)

        # Tools
        self.bridge = CvBridge()

    def runner(self, camera_data):
        """
        callback function for image subscriber, every frame gets scanned for board and publishes to board_center topic
        (for robot movement) and board tile centers (for game state updates)
        :param camera_data: Camera data input from subscriber
        """
        try:
            # ToDo: Check if this works. Or default back to [640,360] (only highlighted in PyCharm)
            boardCenter = [camera_data.width / 2, camera_data.height / 2]  # Initialize as center of frame

            # Convert Image to CV2 Frame
            cv_image = self.bridge.imgmsg_to_cv2(camera_data, "bgr8")
            boardImage = cv_image.copy()

            # characterize board location and orientation
            scaledCenter, boardImage, tf_board2camera = detectBoard(cv_image)


            # For visual purposes, simple crop
            # cropped_image = frame[center[1]-90:center[1]+90,center[0]-90:center[0]+90] #640x480
            # cropped_image = frame[center[1] - 125:center[1] + 125, center[0] - 125:center[0] + 125]  # 1280x720

            # find all 9 nine tile centers based on board center
            tileCentersMatrices = define_board_tile_centers()

            tileCenters2camera = tf.convertPath2FixedFrame(tileCentersMatrices, tf_board2camera)  # 4x4 transformation matrix

            # Columns: 0,1,2 are rotations, column: 3 is translation
            # Rows: 0,1 are x & y rotation & translation values

            import subprocess

            ## Build Camer_2_World TF
            # todo: Rewrite how the camera-tf is defined. Suggestion: grab from topic 'tf'
            ttt_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            tf_camera2world_filepath = np.load(ttt_pkg + "/tf_camera2world.npy")
            tf_camera2world = tf.quant_pose_to_tf_matrix(tf_camera2world_filepath)

            rot_camera_hardcode = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

            translation = tf_camera2world[:-1, -1].tolist()
            tf_camera2world = tf.generateTransMatrix(rot_camera_hardcode, translation)

            ## Build Board_2_World TF
            tf_board2world = np.matmul(tf_camera2world, tf_board2camera)

            ## Convert TF (4x4) Array to Pose (7-element list)
            pose_goal = tf.transformToPose(tf_board2world)

            ## Publish Board Pose
            transform_msg = TransformStamped()
            transform_msg.header.frame_id = 'Origin'
            transform_msg.child_frame_id = 'Board'
            transform_msg.transform.translation.x = pose_goal[0]
            transform_msg.transform.translation.y = pose_goal[1]
            transform_msg.transform.translation.z = pose_goal[2]
            transform_msg.transform.rotation.x = pose_goal[3]
            transform_msg.transform.rotation.y = pose_goal[4]
            transform_msg.transform.rotation.z = pose_goal[5]
            transform_msg.transform.rotation.w = pose_goal[6]

            # Publish
            self.center_transform_pub.publish(transform_msg)
            rospy.loginfo(transform_msg)

            # Draw Tile Numbers onto Frame
            xyList = [[] for i in range(9)]
            scale = .895 / 1280  # todo: set to camera intrinsics
            for i in range(9):
                tileTranslationMatrix = (tileCenters2camera[i][0:2, 3:4])  # in cm
                x = tileTranslationMatrix[0] / scale + 640
                y = tileTranslationMatrix[1] / scale + 360  # in pixels
                xyList[i].append(int(x))
                yyLis[i].append(int(y))
                cv2.putText(boardImage, str(i), (int(xList[i]), int(yList[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 0),
                            2)

            # save pixel locations for tiles
            tictactoe_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            filename = 'tile_centers_pixel.npy'

            outputFilePath = tictactoe_pkg + '/' + filename
            np.save(outputFilePath, data_list)

            # Image Stats
            height, width, channels = boardImage.shape

            # Prepare Image Message
            msg_img = Image()
            msg_img.height = height
            msg_img.width = width
            msg_img.encoding = 'rgb8'
            try:
                msg_img.data = self.bridge.cv2_to_imgmsg(boardImage, 'rgb8')
            except CvBridgeError as e:
                print(e)

            # Publish Image with game board visuals
            self.camera_tile_annotation.publish(msg_img)
            cv2.imshow('CV2: Live Board', boardImage)
            cv2.waitKey(3)

            # Publish Transform
            self.center_transform_pub.publish(transform_msg)
            rospy.loginfo(transform_msg)

            # Publish Tile centers
            # todo: not sure what msg to use for tile centers





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
    """

    # Setup Node
    rospy.init_node('board_vision_processor', anonymous=False)
    print(">> Board Vision Processor Node Successfully Created")

    # Listeners
    # Setup Listener to Automatically run the Board_Publisher whenever an image frame is received.
    bp_callback = board_publisher()

    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, bp_callback.runner)

    # Auto-Run until launch file is shutdown
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()