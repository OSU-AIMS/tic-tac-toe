#!/usr/bin/env python

import rospy
from robot_support import *



class anyPosition(object):
  def __init__(self):
    super(anyPosition,self).__init__()
    self.rc = moveManipulator('bot_mh5l_pgn64')
    self.rc.set_vel(0.1)
    self.rc.set_accel(0.1)

  def moveTo(self,x,y):
    pose_lower = [x,y,0.01,0,1,0,0]
    self.rc.goto_Quant_Orient(pose_lower)

  def start_position(self):
    self.rc.goto_all_zeros()
    #self.rc.add_object()

  def scanPos(self):
    joint_goal = self.rc.move_group.get_current_joint_values()

    joint_goal[0] = radians(-90)
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
    p= anyPosition(0,-0.6)
    raw_input("Start <enter>")
    p.start_position()
    p.lineup()
  

  except rospy.ROSInterruptException:
    exit()
  except KeyboardInterrupt:
    exit()


if __name__ == '__main__':
  main()


 
 