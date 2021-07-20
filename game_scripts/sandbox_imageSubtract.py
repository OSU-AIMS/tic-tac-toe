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
    board_precrop = cv2.imread('images/boardwO_Color.png')
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
    cv2.imshow('subtract(background,board)',piece)
    #board_outline = cv2.subtract(board,background)
    #cv2.imshow('subtract(board,background)',board_outline)

    
    # Try Canny Edge detection then run contour detect
    piece_canny = cv2.Canny(piece,100,200)
    cv2.imshow('Canny Piece',piece_canny)


    # Get Contour of Letter 
    # from https://stackoverflow.com/questions/68383697/find-center-coordinates-of-each-character-in-word-on-image-opencv-python
    
    colors = [(123,200,0),(200,0,0), (0,200,0),(0,0,200),(200,0,123),(0,123,200),(200,0,200)]
    #  used to draw circles (but circles not showing up as of 7/20/2021)
    #  200,0,0: red    200,0,123: fushia 
    #  0,200,0: green  0,123,200: teal   123,200,0: light green   
    #  0,0,200: blue   200,0,200: magenta 



    ## Turn images into binary & then try contour detection
    #piece_binary, _= cv2.threshold(piece,15,255,cv2.THRESH_BINARY)
    
    #cv2.imshow('binary',piece_binary)

    #piece_binary =round(piece_binary).astype(int)



    
    #piece = piece.astype(np.uint8) # findcountours only supports image in grayscale uint8 format
    CannyCopy = piece_canny.copy() # creates copy
    piece_canny = piece_canny.astype(np.uint8)
    #cv2.imshow('Piece: uint8',background)
    
    # contours,hierarchy = cv2.findContours(piece, cv2.THRESH_BINARY, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contours,hierarchy = cv2.findContours(piece_canny,cv2.THRESH_BINARY,cv2.CHAIN_APPROX_SIMPLE) [-2:]

    # By using [-2:], we are basically taking the last two values from the tuple returned by cv2.findContours. 
    # Since in some versions, it returns (image, contours, hierarchy) and in other versions it returns...
    # ...(contours, hierarchy), contours, hierarchy are always the last two values.
    
    #print('Hierarchy',hierarchy)
    #print('Contours:',contours)


    for contour in contours:
      cv2.drawContours(CannyCopy,[contour],-1,colors[0],2)
      colors.pop(0)
      cv2.imshow('Drawn Contour on Canny',CannyCopy)


    
    # get centroid (x,y) location 
    for contour in contours:
      center,radius = cv2.minEnclosingCircle(contour)
    print('Center (unit:pixels ?):',center)
    print('Radius (unit:pixels ?):',radius)

    # if center values matches with grid space from Luis's grid creation code, 
    # then you know which square it is in & can update the global board variable then


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




