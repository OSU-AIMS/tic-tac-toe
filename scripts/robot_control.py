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
from motoman_msgs.srv import ReadSingleIO, WriteSingleIO


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
        # rospy.init_node('node_moveManipulator', anonymous=True)

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

        self.move_group.MoveGroupCommander.set_planning_time(20)
        self.move_group.MoveGroupCommander.set_num_planning_attempts(40)

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

    def send_io(self, request):
        ## Wrapper for rosservice to open/close gripper using Read/Write IO

        # Wait for ros services to come up
        rospy.wait_for_service('read_single_io')
        rospy.wait_for_service('write_single_io')

        # Create Handle for Service Proxy's
        try:
            read_single_io = rospy.ServiceProxy('read_single_io', ReadSingleIO)
            write_single_io = rospy.ServiceProxy('write_single_io', WriteSingleIO)
        except rospy.ServiceException as e:
            print("Gripper IO Service Call failed: %s"%e)

        # Send 'Write' IO Message
        try:
          write_status = write_single_io(10010, request)
        except:
          print("An exception occured. Unable to write to Single IO.")


        # Call Read Service to check current position
        read_status = read_single_io(10011).value
        if read_status:
            print('Gripper is Closed')
        else:
            print('Gripper is Open')

        return read_status

    def goto_Pose(self, pose_goal):
        """
            Set pose_goal.
            Plan and execute.
            Stop to prevent residual motion
            Clear goal from target list.
        """
        self.move_group.set_pose_target(pose_goal)
        plan = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def goto_Pose_w_tolerance(self, pose_goal, joint_tol=0, position_tol=0, orientation_tol=0):
        """
            Set pose_goal.
            Plan and execute.
            Stop to prevent residual motion
            Clear goal from target list.
        """
        self.move_group.MoveGroupCommander.set_goal_joint_tolerance(joint_tol)
        self.move_group.MoveGroupCommander.set_goal_orientation_tolerance(position_tol)
        self.move_group.MoveGroupCommander.set_goal_position_tolerance(orientation_tol)

        self.move_group.set_pose_target(pose_goal)
        plan = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()




