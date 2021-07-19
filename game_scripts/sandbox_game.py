#!/usr/bin/env python

# import sys
  
# # adding Folder_2 to the system path
# sys.path.insert(0, '/home/martinez737/tic-tac-toe_ws/src/tic-tac-toe/game_scripts')
# from robot_support import *

from move_piece import *
from scan_board import *
from Realsense_tools import *
import cv2
from tictactoe_brain import *

#PickP = anyPosition()
imgClass = RealsenseTools()
dXO = detectXO()
#brain = BigBrain()

def main():
  try:

    img_precrop = imgClass.grabFrame() 
    #cv2.imshow('Precrop',img_precrop)
    #img = imgClass.cropFrame(img_precrop)
    #cv2.imshow('Cropped Game Board',img)
    #rospy.sleep(5)
    img_gray = cv2.cvtColor(img_precrop,cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',img_gray)
    img_with_contours, BiggestContour, BiggestBounding ,cX,cy = dXO.getContours(img_precrop,img_gray)
    #cv2.imshow('boardContour',img_with_contours)
    #centers = dXO.getCircle(img)
    cv2.waitKey(0)
  except rospy.ROSInterruptException:
    exit()
  except KeyboardInterrupt:
    exit()


if __name__ == '__main__':

  main()

