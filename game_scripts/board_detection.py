#!/usr/bin/env python

# import sys
  
# # adding Folder_2 to the system path
# sys.path.insert(0, '/home/ros/tic-tac-toe_ws/src/tic-tac-toe/edge_scripts')
# #from robot_support import *

from move_piece import *
from scan_board import *

## scan board 

## wait for human move or play move

## scan board and think of best move

### starting X and O in stacked same place (code recurrsive pickup every move with decreasing z values for stack)
## pickup from stack

## play move

def main():
  PickP = anyPosition()
  imgClass = RealsenseTools()
  dXO = detectXO()
  try:
    
    while gameInProgress == True:
      raw_input('Press enter after move:')
      countO = 0
      countX = 1 # if user starts with X -> countX = 1, same with o


      while RoboTurn == True: # its the robot's turn
        ##scanning
        #robot goes into scannning position
        #robot_toPaper
        PickP.scanPos()

        #scan 

        #rectangle detection paper for a 3x3 or just crop
        #scan (letter detection) for (human is X) first X position

        #from x y pixel position find which square its in
        #input into prebuilt tictactoe script
        #recieve output move and execute

        #moveManipulator class
        ## grab from O stack and play move  
        RoboTurn = False






  except rospy.ROSInterruptException:
    exit()
  except KeyboardInterrupt:
    exit()


if __name__ == '__main__':

  main()


 
