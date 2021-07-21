#!/usr/bin/env python

# import sys
# sys.path.insert(0, '/home/martinez737/tic-tac-toe_ws/src/tic-tac-toe/edge_scripts')
# # sys.path.insert(0, '/home/aims-zaphod/tic-tac-toe_ws/src/tic-tac-toe/edge_scripts')

import rospy
from robot_support import *
import math
from math import pi, radians, sqrt
import numpy




class anyPosition(object):
  def __init__(self):
    super(anyPosition,self).__init__()
    self.rc = moveManipulator('bot_mh5l_pgn64')
    self.rc.set_vel(0.1)
    self.rc.set_accel(0.1)

  def moveToPickup(self,x,y):
    
    pose_higher = [x,y,0.05,.707,-.707,0,0]
    self.rc.goto_Quant_Orient(pose_higher)
    raw_input('Lower gripper...')

    pose_lower = [x,y,0.02,.707,-.707,0,0]
    self.rc.goto_Quant_Orient(pose_lower)

    self.rc.send_io(1) #close gripper
    pose_higher = [x,y,0.1,.707,-.707,0,0]
    self.rc.goto_Quant_Orient(pose_higher)

  def moveToBoard(self,x,y):
    pose_higher = [x,y,0.05,.707,-.707,0,0]
    self.rc.goto_Quant_Orient(pose_higher)
    raw_input('Lower gripper...')

    pose_lower = [x,y,0.02,.707,-.707,0,0]
    self.rc.goto_Quant_Orient(pose_lower)

    self.rc.send_io(0) #open gripper
    pose_higher = [x,y,0.1,.707,-.707,0,0]
    self.rc.goto_Quant_Orient(pose_higher)


  def start_position(self):
    self.rc.goto_all_zeros()
    #self.rc.add_object()

  def scanPos(self):
    self.rc.send_io(0) #open grippper
    joint_goal = self.rc.move_group.get_current_joint_values()

    joint_goal[0] = radians(90) #for right side of table
    joint_goal[1] = 0
    joint_goal[2] = 0
    joint_goal[3] = 0
    joint_goal[4] = 0
    joint_goal[5] = 0

    # Send action to move-to defined position
    self.rc.move_group.go(joint_goal, wait=True)

    # Calling ``stop()`` ensures that there is no residual movement
    self.rc.move_group.stop() 

def main():
  try:
    # z_angle = 67.5
    # z_orient=-180
    # z_rot = ([math.cos(radians(z_orient)),-math.sin(radians(z_orient)),0],
    #         [math.sin(radians(z_orient)),math.cos(radians(z_orient)),0],
    #         [0,0,1])
    # y_orient=-180
    # y_rot=([math.cos(radians(y_orient)),0,math.sin(radians(y_orient))],
    #         [0,1,0],
    #         [-math.sin(radians(y_orient)),0,math.cos(radians(y_orient))])
    # x_orient=-90
    # x_rot=([1,0,0],
    #         [0,math.cos(radians(x_orient)), -math.sin(radians(x_orient))],
    #         [0,math.sin(radians(x_orient)),math.cos(radians(x_orient))])

    # camera_rot= numpy.dot(z_rot,y_rot)
    # camera_rotMatrix=numpy.dot(camera_rot,x_rot)


    # z_twist = ([math.cos(z_angle),-math.sin(z_angle),0],
    #           [math.sin(z_angle),math.cos(z_angle),0],                                                                                                                                                  
    #           [0,0,1])

    # rot_twist = numpy.dot(y_rot,z_twist)
    # print(rot_twist)

    # t= [rot_twist[0,0],rot_twist[0,1],rot_twist[0,2],rot_twist[1,0],rot_twist[1,1],rot_twist[1,2],rot_twist[2,0], rot_twist[2,1], rot_twist[2,2]]
    # #matrix to quat
    # w = sqrt(t[0]+t[4]+t[8]+1)/2
    # x = sqrt(t[0]-t[4]-t[8]+1)/2
    # y = sqrt(-t[0]+t[4]-t[8]+1)/2
    # z = sqrt(-t[0]-t[4]+t[8]+1)/2
    # a = [w,x,y,z]
    # m = a.index(max(a))
    # if m == 0:
    #     x = (t[7]-t[5])/(4*w)
    #     y = (t[2]-t[6])/(4*w)
    #     z = (t[3]-t[1])/(4*w)
    # if m == 1:
    #     w = (t[7]-t[5])/(4*x)
    #     y = (t[1]+t[3])/(4*x)
    #     z = (t[6]+t[2])/(4*x)
    # if m == 2:
    #     w = (t[2]-t[6])/(4*y)
    #     x = (t[1]+t[3])/(4*y)
    #     z = (t[5]+t[7])/(4*y)
    # if m == 3:
    #     w = (t[3]-t[1])/(4*z)
    #     x = (t[6]+t[2])/(4*z)
    #     y = (t[5]+t[7])/(4*z)
    # b = [w,x,y,z]
    # print(b)

    
    rc = moveManipulator('bot_mh5l_pgn64')
    rc.set_vel(0.1)
    rc.set_accel(0.1)

    rc.send_io(0) #open grippper
    # pose_higher = [-.12,.664,0.04, 0.707, -0.707, 0,0]
    # rc.goto_Quant_Orient(pose_higher)

  except rospy.ROSInterruptException:
    exit()
  except KeyboardInterrupt:
    exit()


if __name__ == '__main__':
  main()


 
 