#!/usr/bin/env python

import sys
import copy
import rospy
import numpy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import math
from math import pi, radians, sqrt
from std_msgs.msg import *
from moveit_commander.conversions import pose_to_list
from motoman_msgs.srv import ReadSingleIO, WriteSingleIO
from geometry_msgs.msg import Pose

## Quaternion Tools
from tf.transformations import euler_from_quaternion, quaternion_from_euler

#from robot_support import moveManipulator
from robot_support import *


def main():
  try:
    rc = moveManipulator('bot_mh5l_pgn64')
    rc.set_vel(0.1)
    rc.set_accel(0.1)

    #starting position and open gripper
    raw_input('Go to All Zeroes <enter>')
    rc.goto_all_zeros()
   
   ######
    joint_goal = rc.move_group.get_current_joint_values()

    joint_goal[0] = radians(90)
    joint_goal[1] = 0
    joint_goal[2] = 0
    joint_goal[3] = 0
    joint_goal[4] = 0
    joint_goal[5] = 0

    # Send action to move-to defined position
    rc.move_group.go(joint_goal, wait=True)

    # Calling ``stop()`` ensures that there is no residual movement
    rc.move_group.stop()   
  except rospy.ROSInterruptException:
    exit()
  except KeyboardInterrupt:
    exit()

if __name__ == '__main__':
  main()