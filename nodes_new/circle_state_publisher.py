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

ttt_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_2_game_scripts = ttt_pkg + '/game_scripts'
sys.path.insert(1, path_2_game_scripts)

import rospy

# ROS Data Types
from std_msgs.msg import ByteMultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

# Custom Tools
# from Realsense_tools import *
from transformations import *
from shape_detector import *
from cv_bridge import CvBridge, CvBridgeError

# System Tools
import time
from math import pi, radians, sqrt

# Ref
# http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20listener%20%28Python%29
# http://docs.ros.org/en/jade/api/geometry_msgs/html/msg/Transform.html
tf = transformations()
shapeDetect = ShapeDetector()


def findDis(pt1x, pt1y, pt2x, pt2y):
    # print('Entered Rectangle_support: findDis function')
    x1 = float(pt1x)
    x2 = float(pt2x)
    y1 = float(pt1y)
    y2 = float(pt2y)
    dis = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (0.5)
    return dis


class circle_state_publisher():
    """
     Custom tictactoe publisher class that finds circles on image and identifies if/where the circles are on the board.
    """

    def __init__(self, circle_state_annotation, circle_board_state):

        # Inputs

        self.circle_state_annotation = circle_state_annotation
        self.circle_board_state = circle_board_state
        # camera_tile_annotation: publishes the numbers & arrows displayed on the image

        # Tools
        self.bridge = CvBridge()

    def runner(self, data):
        """
        Callback function for image subscriber
        :param data: Camera data input from subscriber
        """
        try:
            board = [0,0,0,0,0,0,0,0,0]
            # array for game board 0 -> empty tile, 1 -> X, -1 -> O
            
            # Convert Image to CV2 Frame
            cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
            img = cv_image.copy()

            centers = shapeDetect.detectCircle(img, radius=15, tolerance=5)

            tictactoe_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            tf_filename = 'tile_centers_pixel.npy'
            xyList = np.load(tictactoe_pkg + '/' + tf_filename)

                
            
            closest_square = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            

            # for each circle found
            for i in range(len(centers)):
                distanceFromCenter = findDis(centers[i][0], centers[i][1], xyList[4][0], xyList[4][1])

                if distanceFromCenter < 145:  # 100 * sqrt2
                    closest_index = None
                    closest = 10000
                    for j in range(9):
                        distance = findDis(centers[i][0], centers[i][1], xyList[j][0], xyList[j][1])

                        # print("Circle {} is {} from tile {}".format(i,distance,j))
                        # findDis params :(pt1x,pt1y, pt2x,pt2y)

                        if distance < 40 and distance < closest:
                            # this creates a boundary just outside the ttt board of 40 pixels away from each tile
                            # any circle within this boundary is likely to be detected as a piece in one of the 9 tiles
                            closest = distance
                            closest_index = j
                    
                    if closest_index is not None:
                        board[closest_index] = -1
                        cv2.circle(img, centers[i], 15, (0, 200, 40), 7)

                        

                        print("Circle {} is in tile {}.".format(i, closest_index))
                    else:
                        print("Circle {} is not on the board".format(i))

            # print('Physical Board: ', board)
            # print('Board Computer sees:', boardCode)

            for i in range(9):
                cv2.putText(img, str(i), (int(xyList[i][0]), int(xyList[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 0, 0),
                            2)
            
            try:
                msg_img = self.bridge.cv2_to_imgmsg(img, 'rgb8')
            except CvBridgeError as e:
                print(e)

            # Publish
            self.circle_state_annotation.publish(msg_img)

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

    pub_circle_board_state = rospy.Publisher("circle_board_state", ByteMultiArray, queue_size=20)

    # Setup Listeners
    cs_callback = circle_state_publisher(pub_circle_state_annotation,pub_circle_board_state)
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, cs_callback.runner)

    # Auto-Run until launch file is shutdown
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
