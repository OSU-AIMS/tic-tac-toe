#!/usr/bin/env python3

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

ttt_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_2_scripts = ttt_pkg + '/scripts'
sys.path.insert(1, path_2_scripts)

import rospy
import tf2_ros
import tf2_msgs.msg

# ROS Data Types
from sensor_msgs.msg import Image
import geometry_msgs.msg
# from geometry_msgs import TransformStamped 

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
            scaledCenter, boardImage, tf_camera2board = detectBoard_contours(cv_image)

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