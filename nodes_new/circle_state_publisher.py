#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2022, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: LuisC18


#####################################################
## IMPORTS
import sys
import os

ttt_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_2_nodes = ttt_pkg + '/nodes'
path_2_scripts = ttt_pkg + '/scripts'

sys.path.insert(1, path_2_nodes)
sys.path.insert(1, path_2_scripts)

import rospy
import tf2_ros
import tf2_msgs.msg

# ROS Data Types
from std_msgs.msg import ByteMultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseArray

# Custom Tools
# from Realsense_tools import *
from transformations import *
from toolbox_shape_detector import *
from cv_bridge import CvBridge, CvBridgeError

# System Tools
import time
from math import pi, radians, sqrt

# Ref
# http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20listener%20%28Python%29
# http://docs.ros.org/en/jade/api/geometry_msgs/html/msg/Transform.html

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

def findDis(pt1x, pt1y, pt2x, pt2y):
    # print('Entered Rectangle_support: findDis function')
    x1 = float(pt1x)
    x2 = float(pt2x)
    y1 = float(pt1y)
    y2 = float(pt2y)
    dis = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (0.5)
    return dis

def prepare_tiles():
    # Values for 3D printed tictactoe board
    centerxDist = 0.0635
    # centeryDist = -0.0635
    centeryDist = 0.0635 
    # removing negative sign fixed board variable, so computer correctly stores location of O blocks

    pieceHeight = 0.03

    """
    tictactoe board order assignment:
    [0 1 2]
    [3 4 5]
    [6 7 8]
    """ 
    tf = transformations()
    # centers =[[-centerxDist ,centeryDist,pieceHeight],[0,centeryDist,pieceHeight],[centerxDist,centeryDist,pieceHeight],
    #                   [-centerxDist,0,pieceHeight],[0,0,pieceHeight],[centerxDist,0,pieceHeight],
    #                   [-centerxDist,-centeryDist,pieceHeight],[0,-centeryDist,pieceHeight],[centerxDist,-centeryDist,pieceHeight]]
    centers =[[centerxDist ,centeryDist,pieceHeight],[0,centeryDist,pieceHeight],[-centerxDist,centeryDist,pieceHeight],
                        [centerxDist,0,pieceHeight],[0,0,pieceHeight],[-centerxDist,0,pieceHeight],
                        [centerxDist,-centeryDist,pieceHeight],[0,-centeryDist,pieceHeight],[-centerxDist,-centeryDist,pieceHeight]]
    #^ puts the +1(X) & -1(O) in the correct spot for the computer to store them.

    tictactoe_center_list = np.array(centers,dtype=np.float)
    rot_default = np.identity((3))
    new_list = []

    for vector in tictactoe_center_list:
        item = np.matrix(vector)
        new_list.append( tf.generateTransMatrix(rot_default, item) )

    return new_list

class circle_state_publisher():
    """
     Custom tictactoe publisher class that finds circles on image and identifies if/where the circles are on the board.
    """

    def __init__(self, circle_state_annotation, circle_board_state,camera2circle,tfBuffer):

        # Inputs

        self.circle_state_annotation = circle_state_annotation
        self.circle_board_state = circle_board_state
        self.pub_camera2circle = camera2circle
        self.tfBuffer = tfBuffer
        self.tf = transformations()
        self.shapeDetect = TOOLBOX_SHAPE_DETECTOR()
        # camera_tile_annotation: publishes the numbers & arrows displayed on the image

        # Tools
        self.bridge = CvBridge()

    def runner(self, data):
        """
        Callback function for image subscriber
        :param data: Camera data input from subscriber
        """
        try:

            pose_data = rospy.wait_for_message("tile_locations",PoseArray,timeout = None)

            xyList = [[] for i in range(9)]
            scale = 1.14 / 1280

            # Convert Image to CV2 Frame
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            img = cv_image.copy()

            xList = []
            yList = []

            for j in range(9):
                xList.append(pose_data.poses[j].position.x)
                yList.append(pose_data.poses[j].position.y)

            # print('xList',xList)
            # print('yList',yList)
            board = [0,0,0,0,0,0,0,0,0]
            # array for game board 0 -> empty tile, 1 -> X, -1 -> O
            
            centers, circles_img = self.shapeDetect.detectCircles(img, radius=5, tolerance=10)
           
            try:
                msg_img = self.bridge.cv2_to_imgmsg(circles_img, 'bgr8')
            except CvBridgeError as e:
                print(e)

            self.circle_state_annotation.publish(msg_img)

            scaledCenter = [0, 0]

            scale = 1.14/1280

            circlesXlist = []
            circlesYlist = []

            for i in range(len(centers)):

                scaledCenter[0] = (centers[i][0] - 640) * scale
                scaledCenter[1] = (centers[i][1] - 360) * scale
            
                tileTranslation = np.array(
                [[.8], [-scaledCenter[0]], [-scaledCenter[1]]])  

                y_neg90 = np.array([[ 0,  0, -1],
                                [0,  1,  0],
                                [1,  0,  0]])

                z_neg90 = np.array([[0,1,0],
                                [-1,0,0],
                                [0,0,1]])
                camera_rot = np.dot(y_neg90,z_neg90)

                tileRotationMatrix = np.dot(camera_rot,np.identity((3)))

                # Build new tf matrix

                tf_camera2circle = np.zeros((4, 4))
                tf_camera2circle[0:3, 0:3] = tileRotationMatrix
                tf_camera2circle[0:3, 3:4] = tileTranslation
                tf_camera2circle[3, 3] = 1

                pose_goal = transformToPose(tf_camera2circle)

                ## Publish Board Pose
                camera2circle_msg = TransformStamped()
                camera2circle_msg.header.frame_id = 'camera_link'
                camera2circle_msg.child_frame_id = 'circles {}'.format(i)
                camera2circle_msg.header.stamp = rospy.Time.now()

                camera2circle_msg.transform.translation.x = pose_goal[0]
                camera2circle_msg.transform.translation.y = pose_goal[1]
                camera2circle_msg.transform.translation.z = pose_goal[2]

                camera2circle_msg.transform.rotation.x = pose_goal[3]
                camera2circle_msg.transform.rotation.y = pose_goal[4]
                camera2circle_msg.transform.rotation.z = pose_goal[5]
                camera2circle_msg.transform.rotation.w = pose_goal[6]

                camera2circle_msg = tf2_msgs.msg.TFMessage([camera2circle_msg])
               
                # Publish
                self.pub_camera2circle.publish(camera2circle_msg) 

                robot_link = 'base_link'
                target_link = 'circles {}'.format(i)

                fixed2circle = self.tfBuffer.lookup_transform(robot_link, target_link, rospy.Time())
                circlesXlist.append(fixed2circle.transform.translation.x)
                circlesYlist.append(fixed2circle.transform.translation.y)

            

            for i in range(len(centers)):
                distanceFromCenter = findDis(circlesXlist[i], circlesYlist[i], xList[4], yList[4])

                if distanceFromCenter < .160:  # 100 * sqrt2 -now in meters
                    closest_index = None
                    closest = 100
                    for j in range(9):
                        distance = findDis(circlesXlist[i], circlesYlist[i], xList[j], yList[j])

                        if distance < .60 and distance < closest:
                            # this creates a boundary just outside the ttt board of 40 pixels away from each tile
                            # any circle within this boundary is likely to be detected as a piece in one of the 9 tiles
                            closest = distance
                            closest_index = j
                        # print('closest_index',closest_index)
                    if closest_index is not None:
                        board[closest_index] = -1
                        # cv2.circle(img, centers[i], 15, (0, 200, 40), 7)

                        print("Circle {} is in tile {}.".format(i, closest_index))
                    else:
                        print("Circle {} is not on the board".format(i))
            

            msg_circle = ByteMultiArray()
            msg_circle.data = board
            self.circle_board_state.publish(msg_circle)
            rospy.loginfo(msg_circle)

        except rospy.ROSInterruptException:
            exit()
        except KeyboardInterrupt:
            exit()
        except CvBridgeError as e:
            print(e)


#####################################################
# MAIN
def main():
    """
    Circle Finder.
    This script should only be launched via a launch script.
        circle_state_annotation: draws game board circles on top of camera tile annotation image
        circle_game_state: outputs game state as board code (for o's only)
        TODO: add X detection capabilities

    """

    # Setup Node
    rospy.init_node('circle_state', anonymous=False)
    rospy.loginfo(">> Circle Game State Node Successfully Created")

    # Setup Publishers
    pub_circle_state_annotation = rospy.Publisher("circle_state_annotation", Image, queue_size=20)
    pub_camera2circle = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=20)

    pub_circle_board_state = rospy.Publisher("circle_board_state", ByteMultiArray, queue_size=20)

    # Setup Listeners
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    cs_callback = circle_state_publisher(pub_circle_state_annotation,pub_circle_board_state,pub_camera2circle,tfBuffer)
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, cs_callback.runner)

    # Auto-Run until launch file is shutdown
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
