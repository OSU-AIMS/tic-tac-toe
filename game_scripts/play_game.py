#!/usr/bin/env python

# import sys
  
# # adding Folder_2 to the system path
# sys.path.insert(0, '/home/ros/tic-tac-toe_ws/src/tic-tac-toe/edge_scripts')
  #from robot_support import *

from move_piece import *
from scan_board import *
from Realsense_tools import *
import cv2
from tictactoe_brain import *

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
  brain = BigBrain()
  board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ] # array for game board 
  boardCode = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ] # array for code to know which player went whree
    # Human: +1 (circles)
    # Computer: -1 (X's)
    # board filled with -1 & +1

  try:
    gameInProgress = True
    countO = 0
    countX = 1 # if user starts with X -> countX = 1, same with o
    
    while gameInProgress == True:
      
      raw_input('Press enter after Player move:')
      
      RoboTurn = True
      while RoboTurn == True: # its the robot's turn
        ##scanning
        #robot goes into scannning position
        #robot_toPaper
        PickP.scanPos()
        img_precrop = imgClass.grabFrame() 
        cv2.imshow('Precrop',img_precrop)
        img = imgClass.cropFrame(img_precrop)
        cv2.imshow('Cropped Game Board',img)
        #rospy.sleep(5)
        centers = dXO.getCircle(img)
        ''' 
        Top left square: (x,y) Pixels
          - top left corner:(7,13)
          - bottom right: (54,60)
        Top middle square: (x,y)
          - top left: (62,12)
          - bottom right: (109,57)
        Top right square: (x,y)
          - top left:(115,12)
          - bottom right: (164,57)

        Mid Left square: (x,y)
          - top left: (6,68)
          - bottom right: (55,111)
        Mid Middle square: (x,y)
          - top left: (62,66)
          - bottom right: (108,109)
        Mid right square: (x,y)
          - top left: (114,64)
          - bottom right: (164,109) 

        Bottom left square: (x,y)
          - top left: (5,118)
          - bottom right: (56,165)
        Bottom mid square: (x,y)
          - top left: (63,119)
          - bottom right: (108,165)
        Bottom right square: (x,y)
          - top left: (115,117)
          - bottom right: (167,163)
          '''
        #print('First array: of circles',centers[0,:])
        #print('First x of 1st array:',centers[1][0])
        #print('Length of CentersLIst',len(centers))
        #  #length = 5 for max
        for i in range(len(centers)):
          # print('i:',i) # starts at 0
          if 7 < centers[i][0] < 54 and 13 < centers[i][1] < 60:
            print('aT top left square')
            board[0][0]='O'
            boardCode[0][0]= +1

          elif 62 < centers[i][0] < 109 and 12 <= centers[i][1] < 57:
            print('At top middle')
            board[0][1]='O'
            boardCode[0][1]= +1

          elif 115 < centers[i][0] < 164 and 12 <= centers[i][1] < 57:
            print('At top right')
            board[0][2]='O'
            boardCode[0][1]= +1

          elif 6 < centers[i][0] < 55 and 68 <= centers[i][1] < 111:
            print('At mid left')
            board[1][0]='O'
            boardCode[1][0]= +1

          elif 62 < centers[i][0] <= 108 and 66 < centers[i][1] < 109:
            print('At mid middle')
            board[1][1]='O'
            boardCode[1][1]= +1

          elif 114 < centers[i][0] <= 164 and 64 < centers[i][1] < 109:
            print('At mid right')
            board[1][2]='O'
            boardCode[1][2]= +1

          elif 5 <= centers[i][0] < 56 and 118 <= centers[i][1] < 165:
            print('At bottom left')
            board[2][0]='O'
            boardCode[2][0]= +1

          elif 63 <= centers[i][0] <= 108 and 19 < centers[i][1] < 165:
            print('At bottom middle')
            board[2][1]= 'O'
            boardCode[2][1]= +1

          elif 115 < centers[i][0] <= 167 and 117 < centers[i][1] < 163:
            print('At bottom right')
            board[2][2]='O'
            boardCode[2][2]= +1


          else:
            print('not on board')
            # 

         
 

        #scan 

        #rectangle detection paper for a 3x3 or just crop
        #scan (letter detection) for (human is X) first X position

        #from x y pixel position find which square its in

        #input into prebuilt tictactoe script
        # ai_turn parameters:(self,c_choice, h_choice,board):
        move = brain.ai_turn('X','O' , boardCode) #outputs move array based on minimx
        print('MOVE: ',move)

        board[move[0]][move[1]]='X'
        boardCode[move[0]][move[1]]= -1



        #recieve output move and execute robot motion

        #moveManipulator class
        ## grab from O stack and play move  
        cv2.waitKey(0)
        RoboTurn = False
      print('Current State of Physical Board:',board) # after robot turn is over  
      print('Current State of Code Board',boardCode)


      # gameInProgress = False






  except rospy.ROSInterruptException:
    exit()
  except KeyboardInterrupt:
    exit()


if __name__ == '__main__':

  main()


 
