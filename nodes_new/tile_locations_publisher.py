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

import rospy
import tf2_ros
import tf2_msgs.msg

# ROS Data Types
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray

def prepare_tiles():
    """
    Adam Buynak's Convenience Function to Convert Path from a List of xyz points to Transformation Matrices
    @return: List of Transformation Matrices
    """
    
    # Values for 3D printed tictactoe board
    centerxDist = 0.0635
    centeryDist = -0.0635

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
        item = np.matrix(vector)
        new_list.append( tf.generateTransMatrix(rot_default, item) )

    return new_list

class tile_locations_publisher():
    """
     Custom tictactoe publisher class that finds circles on image and identifies if/where the circles are on the board.
    """

    def __init__(self, tile_annotation, tile_locations, tfBuffer, tf):

        # Inputs

        self.tile_annotation = tile_annotation
        self.tile_locations = tile_locations
        self.tfBuffer = tfBuffer
        # Tools
        self.bridge = CvBridge()
        self.tf = tf

    def runner(self,data):

        fixed2board = self.tfBuffer.lookup_transform('base_link', 'ttt_board', rospy.Time(0))
        
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

        tf_fixed2board = self.tf.quant_pose_to_tf_matrix(quant_board2world)

        tf_fixed2tiles = self.tf.convertPath2FixedFrame(matrix_tile_centers, tf_fixed2board)

        matr_rot = tileCenters2world[0][0:3, 0:3]

        b = Quaternion(matrix=matr_rot)

        for i in range(9):
            trans_rot = tileCenters2world[i][0:3, 3:4]
            new_pose = [trans_rot[0][0], trans_rot[1][0], trans_rot[2][0], .707, -.707, 0, 0]
            robot_poses.append(new_pose)

        poses_msg = geometry_msgs.msg.PoseArray()
        poses_msg.poses = robot_poses

        self.tile_locations.publish(poses_msg)



        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        boardImage = cv_image.copy()
        
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

        try:
            msg_img = self.bridge.cv2_to_imgmsg(boardImage, 'bgr8')
        except CvBridgeError as e:
            print(e)

        # Publish
        self.tile_annotation.publish(msg_img)



def main():
    """

    """
    tf = transformations()

    # Setup Node
    rospy.init_node('tile_locations', anonymous=False)
    rospy.loginfo(">> Tile Locator Node Successfully Created")

    # Setup Publishers
    pub_tile_annotation = rospy.Publisher("tile_annotation", Image, queue_size=20)

    pub_tile_locations = rospy.Publisher("tile_locations", PoseArray, queue_size=20)

    # Setup Listeners
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    tl_callback = tile_locations_publisher(pub_tile_annotation, pub_tile_locations, tfBuffer,tf)
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, tl_callback.runner)

    # Auto-Run until launch file is shutdown
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()