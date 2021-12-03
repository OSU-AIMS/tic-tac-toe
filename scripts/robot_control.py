#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2021, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: acbuynak


import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    all_equal = True
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)

    return True


class moveManipulator(object):
    """moveManipulator Class"""
    def __init__(self, group):
        super(moveManipulator, self).__init__()

        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('node_moveManipulator', anonymous=True)

        # Setup Variables needed for Moveit_Commander
        self.object_name = ''
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = group
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)
        self.group_names = self.robot.get_group_names()

    def set_vel(self,max_vel):
        self.move_group.set_max_velocity_scaling_factor(max_vel)

    def set_accel(self,max_accel):
        self.move_group.set_max_acceleration_scaling_factor(max_accel)

    def lookup_pose(self):
        return self.move_group.get_current_pose(self.eef_link).pose

    def goto_all_zeros(self):
        goto_joint_posn([0, 0, 0, 0, 0, 0])

    def goto_named_target(self, target):
        ## Trajectory in JOINT space

        # Send action to move-to defined position
        self.move_group.set_named_target(target)
        self.move_group.plan()
        self.move_group.go(wait=True)
        self.move_group.stop()

        # For testing:
        current_joints = self.move_group.get_current_joint_values()
        return all_close(target, current_joints, 0.01)

    def goto_joint_posn(self,joint_goal):
        ## Trajectory in JOINT space

        # Motion command w/ residual motion stop
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()

        # For testing:
        current_joints = self.move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)


