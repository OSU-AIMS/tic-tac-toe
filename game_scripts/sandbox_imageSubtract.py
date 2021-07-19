#!/usr/bin/env python

#Zaphod: Python version: 2.7.17




######################################################################################################


# import sys
  
# # adding Folder_2 to the system path
# sys.path.insert(0, '/home/martinez737/tic-tac-toe_ws/src/tic-tac-toe/game_scripts')
# sys.path.insert(0,'/home/aims-zaphod/tic-tac-toe_ws/src/tic-tac-toe/game_scripts')
# sys.path.insert(0,'/home/aims-zaphod/tic-tac-toe_ws/src/tic-tac-toe/edge_scripts')
# sys.path.insert(0,'/home/aims-zaphod/tic-tac-toe_ws/src/tic-tac-toe/color_scripts')
# from robot_support import *

# from rectangle_support import *
# from move_piece import *
# from scan_board import *
# from Realsense_tools import *
import cv2
import rospy
import numpy as np
# from tictactoe_brain import *

# PickP = anyPosition()
# imgClass = RealsenseTools()
# dXO = detectXO()
# brain = BigBrain()
# rect = detectRect()


# importing pyTessarct
# source: https://towardsdatascience.com/read-text-from-image-with-one-line-of-python-code-c22ede074cac


def main():
  try:
    print('OpenCV version:',cv2.__version__)
    #Marvin: 
    #Zaphod: 3.2
    #Scott Labs:4.2

    # load images & crop them
    background_precrop = cv2.imread('images/board_Color.png')
    background_crop = background_precrop[0:300, 10:550] #(y,x)
    # above crop removes floor & focuses on table around paper
    #cv2.imshow('Pre-Crop',background_precrop)
    #cv2.imshow('Should be only table, no ground',background_crop)
    board_precrop = cv2.imread('images/boardwX_Color.png')
    board_crop = board_precrop[0:300,10:550]
    
    # create grayscale images
    background = cv2.cvtColor(background_crop,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Grayscale of cropped image',background)
    board = cv2.cvtColor(board_crop,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Grayscale of cropped board',board)

    # Subtract them
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    background = cv2.morphologyEx(background, cv2.MORPH_ERODE, kernel,iterations = 5)
    piece = cv2.subtract(background,board)
    cv2.imshow('subtract(background,board) Subtracted gray scale. Should be 1 spot standing out',piece)
    board_outline = cv2.subtract(board,background)
    cv2.imshow('subtract(board,background)',board_outline)

    # Get Contour of Letter 
    # from https://stackoverflow.com/questions/68383697/find-center-coordinates-of-each-character-in-word-on-image-opencv-python
    
    colors = [(200,0,0), (0,200,0),(0,0,200),(123,200,0),(200,0,123),(0,123,200),(200,0,200)]

    ## Turn images into binary & then try contour detectoin *&^(*&(*))
    
    background = background.astype(np.uint8) #findcountours only supports image in grayscale uint8 format
    cv2.imshow('Piece: uint8',background)
    
    contours,hierarchy = cv2.findContours(background, cv2.THRESH_BINARY, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # By using [-2:], we are basically taking the last two values from the tuple returned by cv2.findContours. 
    # Since in some versions, it returns (image, contours, hierarchy) and in other versions it returns...
    # ...(contours, hierarchy), contours, hierarchy are always the last two values.
    
    # print('Contours:',contours)


    for contour in contours:
      cv2.drawContours(background,[contour],-1,colors[0],2)
      colors.pop(0)

    
    # get centroid (x,y) location
    for contour in contours:
      center,_ = cv2.minEnclosingCircle(contour)



    # img_precrop = imgClass.grabFrame() 
    # #cv2.imshow('Precrop',img_precrop)
    # #img = imgClass.cropFrame(img_precrop)
    # #cv2.imshow('Cropped Game Board',img)
    # #rospy.sleep(5)
    # img_gray = cv2.cvtColor(img_precrop,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',img_gray)
    # img_with_contours, BiggestContour, BiggestBounding ,cX,cy = dXO.getContours(img_precrop,img_gray)
    # #cv2.imshow('boardContour',img_with_contours)
    # #centers = dXO.getCircle(img)
    cv2.waitKey(0)
  except rospy.ROSInterruptException:
    exit()
  except KeyboardInterrupt:
    exit()


if __name__ == '__main__':

  main()




