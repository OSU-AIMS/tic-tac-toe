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
import numpy as np
#PickP = anyPosition()
imgClass = RealsenseTools()
dXO = detectXO()
#brain = BigBrain()
def detectBoard():
  boardCenter=[0,0]
  while boardCenter[0]==0:
    full_frame = imgClass.grabFrame()

    # small crop to just table
    table_frame =full_frame[0:480,0:570] # frame[y,x]

    boardImage, boardCenter, boardPoints= dXO.getContours(table_frame)

    print(boardCenter)
    
  return boardImage, boardCenter, boardPoints

def main():
  try:
    boardImage, boardCenter,boardPoints = detectBoard()
    #centers = dXO.getCircle(img)
    angle = dXO.getOrientation(boardPoints, boardImage)
    print(np.rad2deg(angle))
    cv2.imshow('board angle',boardImage)


    img = imgClass.grabFrame()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges= imgBlur.copy()
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 3)
    kernel = np.ones((5,5),np.uint8)
    edges = cv2.erode(edges,kernel,iterations = 2)
    edges = cv2.Canny(edges,100,200,apertureSize = 3)
    
    #cv2.imwrite('canny.jpg',edges)

    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    cv2.imshow('edges',edges)
    cv2.waitKey(0)
    #if not lines.any():
      #

    if filter:
      rho_threshold = 15
      theta_threshold = 0.1

      # how many lines are similar to a given one
      similar_lines = {i : [] for i in range(len(lines))}
      for i in range(len(lines)):
        for j in range(len(lines)):
          if i == j:
            continue

          rho_i,theta_i = lines[i][0]
          rho_j,theta_j = lines[j][0]
          if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
            similar_lines[i].append(j)

      # ordering the INDECES of the lines by how many are similar to them
      indices = [i for i in range(len(lines))]
      indices.sort(key=lambda x : len(similar_lines[x]))

      # line flags is the base for the filtering
      line_flags = len(lines)*[True]
      for i in range(len(lines) - 1):
        if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
          continue

        for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
          if not line_flags[indices[j]]: # and only if we have not disregarded them already
            continue

          rho_i,theta_i = lines[indices[i]][0]
          rho_j,theta_j = lines[indices[j]][0]
          if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
            line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

    print('number of Hough lines:', len(lines))

    filtered_lines = []

    if filter:
      for i in range(len(lines)): # filtering
        if line_flags[i]:
          filtered_lines.append(lines[i])

      print('Number of filtered lines:', len(filtered_lines))
    else:
      filtered_lines = lines

    for line in filtered_lines:
      rho,theta = line[0]
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 1000*(-b))
      y1 = int(y0 + 1000*(a))
      x2 = int(x0 - 1000*(-b))
      y2 = int(y0 - 1000*(a))

      cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow('grid lines',img)
    cv2.waitKey(0)
  except rospy.ROSInterruptException:
    exit()
  except KeyboardInterrupt:
    exit()


if __name__ == '__main__':

  main()

