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

    def xPickup(self, x_count):
        """
        Executes a pickup command to a specified x,y,z location (with respect to robot origin)
        Default values for z = .1 , hovers at 10cm above origin plane and lowers to 5 cm below input z value.
        :param x: X position in meters
        :param y: Y position in meters
        :param z: Z position in meters.
        """

        x_position = -0.11
        x_index = x_count 
        y_position = -(x_index*0.0354 + .5)
        z_position_hover = 0
        z_position = -.1

        self.openGripper()       
        pose_higher = [x_position, y_position, z_position_hover, .707, -.707, 0, 0]
        self.rc.goto_Quant_Orient(pose_higher)

        pose_lower = [x_position, y_position, z_position, .707, -.707, 0, 0]
        self.rc.goto_Quant_Orient(pose_lower)
        self.closeGripper()             
        
        self.rc.goto_Quant_Orient(pose_higher)

    def placePiece(self,tile_robot_pose):
        tile_robot_pose_hover = tile_robot_pose
        tile_robot_pose_hover.position.z = .1
        self.rc.goto_Pose(tile_robot_pose_hover)

        self.rc.goto_Pose(tile_robot_pose)

        self.openGripper()

        self.rc.goto_Pose(tile_robot_pose_hover)

        
    

    