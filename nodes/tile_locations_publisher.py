#!/usr/bin/env python

#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2022, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: LuisC18

import os
import sys
ttt_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_2_nodes = ttt_pkg + '/nodes'
sys.path.insert(1, path_2_nodes) 

from cv_bridge import CvBridge, CvBridgeError
import cv2

import rospy
import tf2_ros
import tf2_msgs.msg

# ROS Data Types
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from transformations import *

def prepare_tiles():
    """
    Adam Buynak's Convenience Function to Convert Path from a List of xyz points to Transformation Matrices
    @return: List of Transformation Matrices
    """
    
    # Values for 3D printed tictactoe board
    centerxDist = 0.0635
    centeryDist = 0.0635

    pieceHeight = 0.03

    """
    tictactoe board order assignment:
    [0 1 2]
    [3 4 5]
    [6 7 8]
    """ 
    tf = transformations()
    centers =[[-centerxDist ,centeryDist,pieceHeight],[0,centeryDist,pieceHeight],[centerxDist,centeryDist,pieceHeight],
                        [-centerxDist,0,pieceHeight],[0,0,pieceHeight],[centerxDist,0,pieceHeight],
                        [-centerxDist,-centeryDist,pieceHeight],[0,-centeryDist,pieceHeight],[centerxDist,-centeryDist,pieceHeight]]

    tictactoe_center_list = np.array(centers,dtype=np.float)
    # print('tictactoe_center_list:\n',tictactoe_center_list)
    rot_default = np.identity((3))
    new_list = []

    for vector in tictactoe_center_list:
        # print(vector)
        item = np.matrix(vector)
        new_list.append( tf.generateTransMatrix(rot_default, item) )

    return new_list

class tile_locations_publisher():
    """
     Custom tictactoe publisher class that finds circles on image and identifies if/where the circles are on the board.
    """

    def __init__(self,tile_locations, tf):

        # Inputs
        self.tile_locations = tile_locations
        # Tools
        self.bridge = CvBridge()
        self.tf = tf

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

    def runner(self,data):
        robot_link = 'base_link'
        target_link = 'ttt_board'

        fixed2board = self.tfBuffer.lookup_transform(robot_link, target_link, rospy.Time())
            
        fixed2board_pose = [ 
            fixed2board.transform.translation.x,
            fixed2board.transform.translation.y,
            fixed2board.transform.translation.z,
            fixed2board.transform.rotation.w,
            fixed2board.transform.rotation.x,
            fixed2board.transform.rotation.y,
            fixed2board.transform.rotation.z
            ]


        matrix_tile_centers = prepare_tiles()

        tf_fixed2board = self.tf.quant_pose_to_tf_matrix(fixed2board_pose)

        tf_fixed2tiles = self.tf.convertPath2FixedFrame(matrix_tile_centers, tf_fixed2board)
        

        robot_poses = []
        poses_msg = PoseArray()
        poses_msg.header.frame_id = "base_link"
        poses_msg.header.stamp = rospy.Time.now()

        for i in range(9):
            trans_rot = tf_fixed2tiles[i][0:3, 3:4]
            
            pose_msg = Pose()
            pose_msg.position.x = trans_rot[0][0]
            pose_msg.position.y = trans_rot[1][0]
            pose_msg.position.z = trans_rot[2][0]
            pose_msg.orientation.x = .707
            pose_msg.orientation.y = -.707
            pose_msg.orientation.z = 0
            pose_msg.orientation.w = 0

            poses_msg.poses.append(pose_msg)


        self.tile_locations.publish(poses_msg)
        rospy.loginfo(poses_msg)


def main():
    """

    """
    tf = transformations()

    # Setup Node
    rospy.init_node('tile_locations', anonymous=False)
    rospy.loginfo(">> Tile Locator Node Successfully Created")

    # Setup Publishers

    pub_tile_locations = rospy.Publisher("tile_locations", PoseArray, queue_size=20)

    # Setup Listeners
    
    tl_callback = tile_locations_publisher(pub_tile_locations, tf)
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, tl_callback.runner)

    # Auto-Run until launch file is shutdown
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()