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
import subprocess
from robot_support import *
#PickP = anyPosition()
imgClass = RealsenseTools()
timer_wait()
dXO = detectXO()
rc = moveManipulator('bot_mh5l_pgn64')

#brain = BigBrain()
def timer_wait():
  try:
    for remaining in range(3,0,-1):
      sys.stdout.write("\r")
      sys.stdout.write("Updating Frames: {:2d} seconds remaining.".format(remaining))
      sys.stdout.flush()
      time.sleep(1)
    sys.stdout.write("\r|                                                    \n")
  except KeyboardInterrupt:
    sys.exit()

def generateTransMatrix(matr_rotate, matr_translate):
  """
  Convenience Function which accepts two inputs to output a Homogeneous Transformation Matrix
  Intended to function for 3-dimensions frames ONLY
  :param matr_rotate: 3x3 Rotational Matrix
  :param matr_translate: 3x1 Translation Vector (x;y;z)
  :return Homogeneous Transformation Matrix (4x4)
  """

  ## If Translation Matrix is List, Convert
  if type(matr_translate) is list:
    matr_translate = np.matrix(matr_translate)
    #print("Changed translation vector from input 'list' to 'np.matrix'")           #TODO Commented out for debugging

  ## Evaluate Inputs. Check if acceptable size.
  if not matr_rotate.shape == (3, 3):
    raise Exception("Error Generating Transformation Matrix. Incorrectly sized inputs.")
  if not matr_translate.size == 3:
    raise Exception("Error Generating Transformation Matrix. Translation Vector wrong size.")

  ## Reformat Inputs to common shape
  if matr_translate.shape == (1, 3):
    matr_translate = np.transpose(matr_translate)
    #print("Transposed input translation vector")                                   #TODO Commented out for debugging

  ## Build Homogeneous Transformation matrix using reformatted inputs
  # Currently includes flexibility to different sized inputs. Wasted process time, but flexible for future.
  # Assigns bottom right corner as value '1'
  new_transformMatrix = np.zeros((4,4))
  new_transformMatrix[0:0+matr_rotate.shape[0], 0:0+matr_rotate.shape[1]] = matr_rotate
  new_transformMatrix[0:0+matr_translate.shape[0], 3:3+matr_translate.shape[1]] = matr_translate
  new_transformMatrix[new_transformMatrix.shape[0]-1,new_transformMatrix.shape[1]-1] = 1

  ## Return result
  return new_transformMatrix

def quant_pose_to_tf_matrix(quant_pose):
  """
  Covert a quaternion into a full three-dimensional rotation matrix.

  Input
  :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 

  Output
  :return: A 3x3 element matrix representing the full 3D rotation matrix. 
           This rotation matrix converts a point in the local reference 
           frame to a point in the global reference frame.
  """

  # Extract Translation
  r03 = quant_pose[0]
  r13 = quant_pose[1]
  r23 = quant_pose[2]


  # Extract the values from Q
  q0 = quant_pose[3]
  q1 = quant_pose[4]
  q2 = quant_pose[5]
  q3 = quant_pose[6]
   
  # First row of the rotation matrix
  r00 = 2 * (q0 * q0 + q1 * q1) - 1
  r01 = 2 * (q1 * q2 - q0 * q3)
  r02 = 2 * (q1 * q3 + q0 * q2)
   
  # Second row of the rotation matrix
  r10 = 2 * (q1 * q2 + q0 * q3)
  r11 = 2 * (q0 * q0 + q2 * q2) - 1
  r12 = 2 * (q2 * q3 - q0 * q1)
   
  # Third row of the rotation matrix
  r20 = 2 * (q1 * q3 - q0 * q2)
  r21 = 2 * (q2 * q3 + q0 * q1)
  r22 = 2 * (q0 * q0 + q3 * q3) - 1
   
  # 3x3 rotation matrix
  tf_matrix = np.array([ [r00, r01, r02, r03],
                         [r10, r11, r12, r13],
                         [r20, r21, r22, r23],
                         [  0,   0,   0,  1 ]])
                          
  return tf_matrix



def detectBoard():
  boardCenter=[0,0]
  while boardCenter[0]==0:
    full_frame = imgClass.grabFrame()

    # small crop to just table
    table_frame =full_frame[0:480,0:640] # frame[y,x]

    boardImage, boardCenter, boardPoints= dXO.getContours(table_frame)
    scale = .664/640 #(m/pixel)
    ScaledCenter = [0,0]
    ScaledCenter[0] = (boardCenter[0]-320)*scale
    ScaledCenter[1] = (boardCenter[1]-240)*scale
    print("Center of board relative to center of camera (cm):",ScaledCenter)


    
  cv2.waitKey(0)

  return boardImage, boardCenter, boardPoints, ScaledCenter

def transformToPose(transform):
  # Location Vector
  pose_goal=[]
  point = transform
  x,y,z = point[:-1,3]
  x = np.asscalar(x)
  y = np.asscalar(y)
  z = np.asscalar(z)

  # Quant Calculation Support Variables
  # Only find trace for the rotational matrix.
  t = np.trace(point) - point[3,3]
  r = np.sqrt(1+t)

  # Primary Diagonal Elements
  Qxx = point[0,0]
  Qyy = point[1,1]
  Qzz = point[2,2]

  # Quant Calculation
  qx = np.copysign(0.5 * np.sqrt(1 + Qxx - Qyy - Qzz), point[2,1]-point[1,2])
  qy = np.copysign(0.5 * np.sqrt(1 - Qxx + Qyy - Qzz), point[0,2]-point[2,0])
  qz = np.copysign(0.5 * np.sqrt(1 - Qxx - Qyy + Qzz), point[1,0]-point[0,1])
  qw = 0.5*r

  pose_goal=[x, y, z, qx,qy,qz, qw]
  return pose_goal





def main():
  try:
    boardImage, boardCenter,boardPoints, ScaledCenter = detectBoard()
    #centers = dXO.getCircle(img)
    angle = dXO.getOrientation(boardPoints, boardImage)
    print(np.rad2deg(angle))
    cv2.imshow('board angle',boardImage)

    boardCropped = imgClass.croptoBoard(boardImage, boardCenter)
    cv2.imshow('Cropped Board',boardCropped)
    cv2.waitKey(0)
    
    
    boardTranslation = np.array([[ScaledCenter[0]],[ScaledCenter[1]],[0.64]])  ## depth of the table is .64 m
    boardRotation = np.identity((3))
    tf_board = generateTransMatrix(boardRotation,boardTranslation) #tf_body2camera, transform from camera 

    import subprocess

    tf_filename = "tf_camera2world.npy"
    tf_listener = '/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/nodes/tf_origin_camera_subscriber.py'
    subprocess.call([tf_listener, '/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe', tf_filename])

    tf_list = np.load(str('/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe') + '/' + tf_filename)
    tf_camera2world = quant_pose_to_tf_matrix(tf_list)
    #print('tf camera to world:',tf_camera2world)
    rot_camera_hardcode  = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

    translate            = tf_camera2world[:-1,-1].tolist()
    tf_camera2world = generateTransMatrix(rot_camera_hardcode, translate)
    print('camera to robot:')
    print(np.around(tf_camera2world,2))

    tf_board2world = np.matmul(tf_camera2world,tf_board)
    print('board to robot:')
    print(np.around(tf_board2world,2))

    boardCenterPose = transformToPose(tf_board2world)
    rc.set_vel(0.1)
    rc.set_accel(0.1)
    raw_input('Go to board center')
    rc.goto_Quant_Orient(boardCenterPose)
    raw_input('Go to all zeros')
    rc.goto_all_zeros()



    
  except rospy.ROSInterruptException:
    exit()
  except KeyboardInterrupt:
    exit()


if __name__ == '__main__':

  main()




############# hough lines/grid lines 

    #   img = imgClass.grabFrame()
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # imgBlur = cv2.GaussianBlur(gray, (7, 7), 0)
    # edges= imgBlur.copy()
    # kernel = np.ones((3,3),np.uint8)
    # edges = cv2.dilate(edges,kernel,iterations = 3)
    # kernel = np.ones((5,5),np.uint8)
    # edges = cv2.erode(edges,kernel,iterations = 2)
    # edges = cv2.Canny(edges,100,200,apertureSize = 3)
    
    # #cv2.imwrite('canny.jpg',edges)

    # lines = cv2.HoughLines(edges,1,np.pi/180,150)
    # cv2.imshow('edges',edges)
    # cv2.waitKey(0)
    # #if not lines.any():
    #   #

    # if filter:
    #   rho_threshold = 15
    #   theta_threshold = 0.1

    #   # how many lines are similar to a given one
    #   similar_lines = {i : [] for i in range(len(lines))}
    #   for i in range(len(lines)):
    #     for j in range(len(lines)):
    #       if i == j:
    #         continue

    #       rho_i,theta_i = lines[i][0]
    #       rho_j,theta_j = lines[j][0]
    #       if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
    #         similar_lines[i].append(j)

    #   # ordering the INDECES of the lines by how many are similar to them
    #   indices = [i for i in range(len(lines))]
    #   indices.sort(key=lambda x : len(similar_lines[x]))

    #   # line flags is the base for the filtering
    #   line_flags = len(lines)*[True]
    #   for i in range(len(lines) - 1):
    #     if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
    #       continue

    #     for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
    #       if not line_flags[indices[j]]: # and only if we have not disregarded them already
    #         continue

    #       rho_i,theta_i = lines[indices[i]][0]
    #       rho_j,theta_j = lines[indices[j]][0]
    #       if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
    #         line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

    # print('number of Hough lines:', len(lines))

    # filtered_lines = []

    # if filter:
    #   for i in range(len(lines)): # filtering
    #     if line_flags[i]:
    #       filtered_lines.append(lines[i])

    #   print('Number of filtered lines:', len(filtered_lines))
    # else:
    #   filtered_lines = lines

    # for line in filtered_lines:
    #   rho,theta = line[0]
    #   a = np.cos(theta)
    #   b = np.sin(theta)
    #   x0 = a*rho
    #   y0 = b*rho
    #   x1 = int(x0 + 1000*(-b))
    #   y1 = int(y0 + 1000*(a))
    #   x2 = int(x0 - 1000*(-b))
    #   y2 = int(y0 - 1000*(a))

    #   cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    # cv2.imshow('grid lines',img)
    # cv2.waitKey(0)

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
#     src= RL.grabFramegleFrame_color()

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
