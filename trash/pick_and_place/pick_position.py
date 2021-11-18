#!/usr/bin/env python

import rospy
from robot_support import *



class anyPosition(object):
  def __init__(self,x,y):
    super(anyPosition,self).__init__()
    self.rc = moveManipulator('bot_mh5l')
    self.eef = moveManipulator('EEF')
    self.rc.set_vel(0.1)
    self.rc.set_accel(0.1)
    self.eef.set_vel(0.1)
    self.eef.set_accel(0.1)
    self.position = (x,y)

  def lineup(self):
    pose_lower = [self.position[0],self.position[1],0.01,0,1,0,0]
    self.rc.goto_Quant_Orient(pose_lower)


  def start_position(self):
    joint_open = self.eef.move_group.get_current_joint_values()
    joint_open[0] = 0.01
    joint_open[1] = -0.01
    self.eef.goto_joint_posn(joint_open)
    self.rc.goto_all_zeros()
    self.rc.add_object()

  def close_gripper(self):
    joint_closed = self.eef.move_group.get_current_joint_values()
    joint_closed[0] = -0.007 #-0.017 = gripper touching
    joint_closed[1] = 0.007
    self.eef.goto_joint_posn(joint_closed)
    

def main():
  try:
    p= anyPosition(0,0.6)
    raw_input("Start <enter>")
    p.start_position()
    p.lineup()
    p.close_gripper()

  except rospy.ROSInterruptException:
    exit()
  except KeyboardInterrupt:
    exit()


if __name__ == '__main__':
  main()


 
 