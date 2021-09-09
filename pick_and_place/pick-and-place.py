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

def callback(data):
  #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
  # global a 
  
  x = data.position.x
  y = data.position.y
  # a = data.position.x
  # b = data.position.y
  z = data.orientation.z
    
    
def listener():
  with open("/home/martinez737/ws_pick_camera/src/pick-and-place/scripts/Coordinate-angle.txt", "r") as f:
  # with open("/home/ros/ws_pick-and-place/src/pick-and-place/scripts/Coordinate-angle.txt", "r") as f:

# used same code to get gripper to paper
# reuse code to properly orient gripper to pick up object

    file_content = f.read()
    fileList = file_content.splitlines()
    xC = float(fileList[0])/100
    yC = float(fileList[1])/100
    z_angle = radians(float(fileList[2]))
    # z_angle = radians(float(fileList[2])-90)

    # t= [math.cos(z_angle),math.sin(z_angle),0,
    # math.sin(z_angle),-math.cos(z_angle),0,0,0,-1]

  z_orient=-180
  z_rot = ([math.cos(radians(z_orient)),-math.sin(radians(z_orient)),0],
          [math.sin(radians(z_orient)),math.cos(radians(z_orient)),0],
          [0,0,1])
  y_orient=-180
  y_rot=([math.cos(radians(y_orient)),0,math.sin(radians(y_orient))],
          [0,1,0],
          [-math.sin(radians(y_orient)),0,math.cos(radians(y_orient))])
  x_orient=-90
  x_rot=([1,0,0],
          [0,math.cos(radians(x_orient)), -math.sin(radians(x_orient))],
          [0,math.sin(radians(x_orient)),math.cos(radians(x_orient))])

  camera_rot= numpy.dot(z_rot,y_rot)
  camera_rotMatrix=numpy.dot(camera_rot,x_rot)


  z_twist = ([math.cos(z_angle),-math.sin(z_angle),0],
            [math.sin(z_angle),math.cos(z_angle),0],                                                                                                                                                  
            [0,0,1])

  rot_twist = numpy.dot(y_rot,z_twist)
  print(rot_twist)

  t= [rot_twist[0,0],rot_twist[0,1],rot_twist[0,2],rot_twist[1,0],rot_twist[1,1],rot_twist[1,2],rot_twist[2,0], rot_twist[2,1], rot_twist[2,2]]
  #matrix to quat
  w = sqrt(t[0]+t[4]+t[8]+1)/2
  x = sqrt(t[0]-t[4]-t[8]+1)/2
  y = sqrt(-t[0]+t[4]-t[8]+1)/2
  z = sqrt(-t[0]-t[4]+t[8]+1)/2
  a = [w,x,y,z]
  m = a.index(max(a))
  if m == 0:
      x = (t[7]-t[5])/(4*w)
      y = (t[2]-t[6])/(4*w)
      z = (t[3]-t[1])/(4*w)
  if m == 1:
      w = (t[7]-t[5])/(4*x)
      y = (t[1]+t[3])/(4*x)
      z = (t[6]+t[2])/(4*x)
  if m == 2:
      w = (t[2]-t[6])/(4*y)
      x = (t[1]+t[3])/(4*y)
      z = (t[5]+t[7])/(4*y)
  if m == 3:
      w = (t[3]-t[1])/(4*z)
      x = (t[6]+t[2])/(4*z)
      y = (t[5]+t[7])/(4*z)
  b = [w,x,y,z]
  print(b)

  pose_goal = [0,0,0,0,0,0,0]
  pose_goal[0] = xC-0.16
  pose_goal[1] = -yC-0.355 
  # pose_goal[0] = xC-0.040
  # pose_goal[1] = -yC-0.385 # y-distance: Why is this negative xC??????
  # I thought robot base x-axis in RVIZ points in same direction as y-axis of paper so shouldn't it be xC-0.385


  # may need to alter these numbers since we're starting from a different location
  pose_goal[2] = 0.1
  #0.1


  # Test 1: Object oriented at 0: Success
  # Test 2: Oriented at -45: success
  # Test 3: Orientated at -90: Success
  # Test 4: Orient at ~ +22 degrees Success
  # Test 5: Orient at -80 degrees:  Success
  # Using +90 degrees should work for most cases, now up to camera to properly recognize angle

  # pose_goal[3]= 0.990268
  # pose_goal[4]=-0.139173
  # pose_goal[5]=0
  # pose_goal[6]=-0.0000012228

  pose_goal[3]=b[1] #x
  pose_goal[4]=b[2] #y
  pose_goal[5]=b[3] #z
  pose_goal[6]=b[0] #w

  return pose_goal

def main():
  try:
   
   ######
    pose_goal = listener()
    print('posegoal:',pose_goal)
    # print('x position:', a)
    # print('y-position:',b)
    # print('z-orientation:',c)
    
    #set velocity of motion
    rc = moveManipulator('bot_mh5l_pgn64')
    rc.set_vel(0.1)
    rc.set_accel(0.1)

    #starting position and open gripper
    raw_input('Go to All Zeroes <enter>')
    rc.goto_all_zeros()
    rc.send_io(0) # open gripper

    # #for simulation
    # #raw_input('Add cube <enter>')
    # #rc.add_object()
   

    raw_input('Begin Pick and Place <enter>')
    rc.goto_Quant_Orient(pose_goal)
    # move to object position

    raw_input('Lower and Grasp <enter>')
    # # pose [pos: x, y, z, axes:x y z w]
    pose_goal[2] =0.01
    #original0.02
    rc.goto_Quant_Orient(pose_goal)
    
    # # #grasp object
    rc.send_io(1) # closer gripper
    #rc.attach_object()

    # #raise object
    # pose_higher = [xC-0.08,-yC-0.37,.815,0,1,0,0]
    # rc.goto_Quant_Orient(pose_higher)
    raw_input('Grasp <enter>')
    pose_goal[2] = 0.1
    rc.goto_Quant_Orient(pose_goal)
    #lower object
    pose_goal[2] = 0.01
    rc.goto_Quant_Orient(pose_goal)
    rospy.sleep(.5)
    #release object
    rc.send_io(0)
    pose_goal[2] =0.1
    rc.goto_Quant_Orient(pose_goal)
    # rc.detach_object()
    raw_input('Return to start <enter>')
    #return to all zeros
    rc.goto_all_zeros()
    
    rc.remove_object()

  except rospy.ROSInterruptException:
    exit()
  except KeyboardInterrupt:
    exit()

if __name__ == '__main__':
  main()