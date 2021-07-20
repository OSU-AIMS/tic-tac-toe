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

#############################################################################################################################

# #!/usr/bin/env python

# """
# @file hough_lines.py
# @brief This program demonstrates line finding with the Hough transform
# """
# import sys
# import math
# import cv2 as cv
# import numpy as np
# from Realsense_tools import *


# def main(argv):
#     ## [load]
#     RL= RealsenseTools()
#     src= RL.grabFrame()

#     ## [edge_detection]
#     # Edge detection
#     #imgBlur = cv.GaussianBlur(src, (1,1), 0)
#     # cv.imshow('blur',imgBlur)
#     kernel = np.ones((3,3),np.uint8)

#     dilate = cv2.dilate(src,kernel,iterations = 1)
#     # dilate = cv2.erode(dilate,kernel,iterations = 4)
#     # dilate = cv2.dilate(dilate,kernel,iterations = 4)
#     cv.imshow('erode',dilate)
#     dst = cv2.Canny(dilate,100, 200, None, 3)
#     ## [edge_detection]

#     # Copy edges to the images that will display the results in BGR
#     cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
#     cdstP = np.copy(cdst)

#     ## [hough_lines]
#     #  Standard Hough Line Transform
#     lines = cv.HoughLines(dst, 2, np.pi / 180, 250, None, 0, 0)
#     ## [hough_lines]
#     ## [draw_lines]
#     # Draw the lines
#     if lines is not None:
#         for i in range(0, len(lines)):
#             rho = lines[i][0][0]
#             theta = lines[i][0][1]
#             a = math.cos(theta)
#             b = math.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#             pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

#             cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
#     ## [draw_lines]

#     ## [hough_lines_p]
#     # Probabilistic Line Transform
    
#     linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 70, None, 100, 20)
#     print(linesP)
#     ## [hough_lines_p]
#     ## [draw_lines_p]
#     # Draw the lines
#     if linesP is not None:
#         for i in range(0, len(linesP)):
#             l = linesP[i][0]
#             #print(l)
#             cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv.LINE_AA)
#             x1 = l[0]
#             x2 = l[2]
#             y1 = l[1]
#             y2 = l[3]
#             print((l[3]-l[1])/(l[2]-l[0]))
#             # if  175 > ((((x2 - x1)**2 + (y2 - y1)**2)**(0.5))) > 160:
#                 # cv.line(src, (l[0], l[1]), (l[2], l[3]), (255,255,0), 3, cv.LINE_AA)
               

#     ## [draw_lines_p]
#     ## [imshow]
#     # Show results
#     cv.imshow("Source", src)
#     cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
#     cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
#     ## [imshow]
#     ## [exit]
#     # Wait and Exit
#     cv.waitKey()
#     return 0
#     ## [exit]

# if __name__ == "__main__":
#     main(sys.argv[1:])
