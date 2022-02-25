#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2021, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: LuisC18

## IMPORTS
from robot_control import *


class TICTACTOE_MOVEMENT(object):
    """
    Class is a collection of movement and grasp functions.
    """

    def __init__(self):
        self.rc = moveManipulator('mh5l')
        self.rc.set_vel(0.1)
        self.rc.set_accel(0.1)
        pass

    def openGripper(self):
        self.rc.send_io(0)

    def closeGripper(self):
        self.rc.send_io(1)    

    def scanPosition(self):
        # default_scan_joint_goal = ([-90, 0, 0, 0, 0, 0])
        self.rc.goto_named_target('overlook-right')
        self.openGripper()

    def xPickup(self, x_count):
        """
        Executes a pickup command to a specified x,y,z location (with respect to robot origin)
        Default values for z = .1 , hovers at 10cm above origin plane and lowers to 5 cm below input z value.
        :param x: X position in meters
        :param y: Y position in meters
        :param z: Z position in meters.
        """
        #  Translation: [-0.346, 0.112, -0.064]
        x_position = 0.112
        x_index = x_count 
        y_position = -(x_index*0.0381 + .346)
        z_position_hover = -.1
        z_position = -.12

        self.openGripper()       
        pose_higher = [x_position, y_position, z_position_hover, .707, -.707, 0, 0]
        self.rc.goto_Pose(pose_higher)

        pose_lower = [x_position, y_position, z_position, .707, -.707, 0, 0]
        raw_input("For x pickup lower <press enter>")
        self.rc.goto_Pose(pose_lower)
        self.closeGripper()             
        
        self.rc.goto_Pose_w_tolerance(pose_higher, joint_tol=10, position_tol=10, orientation_tol=10)

    def placePiece(self,tile_robot_pose):
        tile_robot_pose_hover = tile_robot_pose
        tile_robot_pose_hover.position.z = 0
        
        self.rc.goto_Pose_w_tolerance(tile_robot_pose_hover, joint_tol=10, position_tol=10, orientation_tol=10)

        raw_input("For x place lower <press enter>")
        tile_robot_pose.position.z = -.13 # originally -0.08 - a bit too high. Piece bounced when dropped which can cause mis-detection of O
        self.rc.goto_Pose_w_tolerance(tile_robot_pose, joint_tol=10, position_tol=10, orientation_tol=10)

        self.openGripper()

        self.rc.goto_Pose_w_tolerance(tile_robot_pose_hover, joint_tol=10, position_tol=10, orientation_tol=10)


        
    

    