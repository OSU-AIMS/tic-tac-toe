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

# ROS Data Types
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

# Custom Tools
  # from Realsense_tools import *
from transformations import *
from scan_board import *
from cv_bridge import CvBridge, CvBridgeError

# System Tools
import time
from math import pi, radians, sqrt

# Ref
# http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20listener%20%28Python%29
# http://docs.ros.org/en/jade/api/geometry_msgs/html/msg/Transform.html
tf = transformations()
dXO = detectXO()



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
    tf = transformations()
    for vector in tictactoe_center_list:
        item = np.matrix(vector)
        tile_locations_tf.append( tf.generateTransMatrix(rot_default, item) )

    return tile_locations_tf


def croptoBoard(frame,center):
    #print('Entered RealsenseTools: cropFrame function\n')
    #cropped_image = frame[55:228,335:515] # frame[y,x]
    # cropped_image = frame[45:218,315:495 ] # frame[y,x]
    # cropped_image = frame[center[1]-90:center[1]+90,center[0]-90:center[0]+90] #640x480
    cropped_image = frame[center[1]-125:center[1]+125,center[0]-125:center[0]+125] #1280x720
    return cropped_image



#####################################################
## PRIMARY CLASS
##

class board_publisher():

    def __init__(self, center_pub, camera_tile_annotation):

        # Inputs
        self.center_pub = center_pub
        self.camera_tile_annotation = camera_tile_annotation

        # Tools
        self.bridge = CvBridge()

    def detectBoard_coloredSquares(self,frame):
        # purpose: recognize  of board based on 3 colored equares


        # import image frame with colored squares
        
        # frame=cv2.imread('../sample_content/sample_images/1X_1O_ATTACHED_coloredSquares_Color_Color.png')
        cv2.imshow('Sample Image',frame)

        # covnert to Grayscale
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscaled Image',gray_frame)

        # Set Thresholds for Red, Green, Blue for detection
        
        



        # Find centers for the squares based on location (ScaledCenter)
        # Look in maze runner d3_post_process



        # Get distance from each center to get board size & center


        # get angle from red to green & use that as the orientation
        # X-vector: Blue --> Red
        # Y-vector: Blue --> Green


        


    def detectBoard(self, frame):
        #table_frame = frame.copy()

        # Reading in static image
        #frame=cv2.imread('../sample_content/sample_images/1X_1O_ATTACHED_coloredSquares_Color_Color.png')
        #cv2.imshow('Sample Image',frame)
        table_frame = frame.copy()


        # cv2.imshow('test',frame)
        # cv2.waitKey(0)
        # small crop to just table
        # table_frame =full_frame[0:480,0:640] # frame[y,x]
        boardImage, boardCenter, boardPoints = dXO.getContours(table_frame)
        # scale = .664/640 #(m/pixel)
        scale = .895 / 1280
        ScaledCenter = [0, 0]
        # ScaledCenter[0] = (boardCenter[0]-320)*scale
        # ScaledCenter[1] = (boardCenter[1]-240)*scale
        ScaledCenter[0] = (boardCenter[0] - 640) * scale
        ScaledCenter[1] = (boardCenter[1] - 360) * scale
        # print("Center of board relative to center of robot (cm):",ScaledCenter)

        return ScaledCenter



    def runner(self, data):
        """
        :param camera_data: Camera data input from subscriber
        """
        try:
            # ToDo: Check if this works. Or default back to [640,360] (only highlighted in PyCharm)
            boardCenter = [data.width/2, data.height/2]   #Initialize as center of frame

            # Convert Image to CV2 Frame
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
            boardImage = cv_image.copy()


            ScaledCenter = self.detectBoard(cv_image)
            #ScaledCenter = self.detectBoard_coloredSquares(cv_image)


            # 
            # print('unordered points:',boardPoints)
            # reorderedPoints = dXO.reorder(boardPoints)
            # # print('reorderedPoints:',reorderedPoints)
            # z_angle = dXO.newOrientation(reorderedPoints)
            #
            # angle = dXO.getOrientation(boardPoints, boardImage)
            # # print('old orientation angle',np.rad2deg(angle))
            #
            # # boardCropped = croptoBoard(boardImage, boardCenter)
            # # print(boardCropped.sh)
            # # cv2.imshow('Cropped Board',boardCropped)

            boardTranslation = np.array(
                [[ScaledCenter[0]], [ScaledCenter[1]], [0.655]])  ## depth of the table is .64 m

            # z_orient = z_angle
            # boardRotation = np.array([[math.cos(radians(z_orient)), -math.sin(radians(z_orient)), 0],
            #                           [math.sin(radians(z_orient)), math.cos(radians(z_orient)), 0],
            #                           [0, 0, 1]])



            # Transformations (board to -imageFrame, -camera, -world)
            tf_board = tf.generateTransMatrix(np.identity((3)), boardTranslation)  # tf_body2camera, transform from camera
            tileCentersMatrices = define_board_tile_centers()
            tileCenters2camera = tf.convertPath2FixedFrame(tileCentersMatrices, tf_board)  # 4x4 transformation matrix

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
            tf_board2world = np.matmul(tf_camera2world, tf_board)

            ## Convert TF (4x4) Array to Pose (7-element list)
            pose_goal = tf.transformToPose(tf_board2world)

            ## Publish Board Pose
            msg = geometry_msgs.msg.TransformStamped()
            msg.header.frame_id = 'Origin'
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
            #rospy.loginfo(msg)




            # Draw Tile Numbers onto Frame
            xList = []
            yList = []
            scale = .895 / 1280  # todo: set to camera intrinsics
            for i in range(9):
                xyzCm = (tileCenters2camera[i][0:2, 3:4])  # in cm
                x = xyzCm[0] / scale + 640
                y = xyzCm[1] / scale + 360  # in pixels
                xList.append(int(x))
                yList.append(int(y))
                cv2.putText(boardImage, str(i), (int(xList[i]), int(yList[i])), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 0),
                            2)

            # Image Stats
            height, width, channels = boardImage.shape

            # Prepare Image Message
            # msg_img = Image()
            # msg_img.height = height
            # msg_img.width = width
            # msg_img.encoding = 'rgb8'
            try:
                msg_img = self.bridge.cv2_to_imgmsg(boardImage, 'rgb8')
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

