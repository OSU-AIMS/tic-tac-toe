#!/usr/bin/env python

#import pyrealsense2 as rs
import numpy as np
import math
import rospy
#from std_msgs.msg import *
import geometry_msgs.msg 
import cv2
#import imutils
#include <opencv2/core.hpp> from docs on PCACompute https://docs.opencv.org/4.2.0/d2/de8/group__core__array.html#ga0ad1147fbcdb256f2e14ae2bfb8c991d

class detectXO(object):
  def __init__(self):
    super(detectXO,self).__init__()
    

  def getCircle(self,frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    rows = blur.shape[0]
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=30)
    centerList = []

    if circles is not None:
      circles = np.uint16(np.around(circles))
      for i in circles[0, :]:
        center = (i[0], i[1]) # assuming (i[0]=x, i[1]=y)
        # circle center
        cv2.circle(frame, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv2.circle(frame, center, radius, (255, 0, 255), 3)
        centerList.append(center)
    
    cv2.imshow("detected circles", frame)
    # cv2.waitKey(0)
    #print("Center List:",centerList)
    # circles outputs (x,y, radius)



    return centerList





    # Reorders the four points of the rectangle
  def reorder(self,points):
    try:
      print('Entered Rectangle_support: reorder function')
      NewPoints = np.zeros_like(points)
      points = points.reshape((4, 2))
      add = points.sum(1)
      NewPoints[0] = points[np.argmin(add)]
      NewPoints[3] = points[np.argmax(add)]
      diff = np.diff(points, axis=1)
      NewPoints[1] = points[np.argmin(diff)]
      NewPoints[2] = points[np.argmax(diff)]
      return NewPoints
    except:
      return []

  # # Finds the distance between 2 points (distance formula)
  # def findDis(self,pts1, pts2):
  #   print('Entered Rectangle_support: findDis function')
  #   x1 = float(pts1[0])
  #   x2 = float(pts2[0])
  #   y1 = float(pts1[1])
  #   y2 = float(pts2[1])
  #   dis = ((x2 - x1)**2 + (y2 - y1)**2)**(0.5)
  #   return dis

  # Draws x-y axis relative to object center and orientation
  def drawAxis(self,img, p_, q_, color, scale):
    print('Entered Rectangle_support: drawAxis function')
    p = list(p_)
    q = list(q_)

    angle = math.atan2(p[1] - q[1], p[0] - q[0])  #in radians
    hypotenuse = math.sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Lengthens the arrows by a factor of scale
    q[0] = p[0] - scale * hypotenuse * math.cos(angle)
    q[1] = p[1] - scale * hypotenuse * math.sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    # Creates the arrow hooks
    p[0] = q[0] + 9 * math.cos(angle + math.pi / 4)
    p[1] = q[1] + 9 * math.sin(angle + math.pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    p[0] = q[0] + 9 * math.cos(angle - math.pi / 4)
    p[1] = q[1] + 9 * math.sin(angle - math.pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

  # Gets angle of object
  def getOrientation(self,pts, img): 
    print('Entered Rectangle_support: getOrientation function')
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
      data_pts[i, 0] = pts[i, 0, 0]
      data_pts[i, 1] = pts[i, 0, 1]

    # Performs PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    #eigenvalues, eigenvectors = np.linalg.eig(mean)


    # Stores the center of the object
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    # Draws the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
          cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
          cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    self.drawAxis(img, cntr, p1, (255, 255, 0), 1)
    self.drawAxis(img, cntr, p2, (0, 0, 255), 5)
    angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    return angle, cntr, mean

    #finds contours on image
  def getContours(self,orignal_frame, gray_mask):
    print('Entered Rectangle_support: getContours function')

    imgBlur = cv2.GaussianBlur(gray_mask, (7, 7), 0)
    #cv2.imshow('blur',imgBlur)
    imgCanny = cv2.Canny(imgBlur, 100, 200)

    kernel = np.ones((3, 3))
    imgDilate = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDilate, kernel, iterations=2)

    #cv2.imshow("img threshold",imgThre)
    contours, _ = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areaList = []
    approxList = []
    centerListX = []
    centerListY = []
    BiggestContour = []
    count = 0
    
    for i in contours:
      print('loop')
      
      #find center using moments
      M = cv2.moments(i)
      centerListX[count] = int((M['m10']/M['m00']))
      centerListY[count] = int((M['m01']/M['m00']))
      #print('Contours: ',contours)

      count += 1
      orignal_copy = orignal_frame.copy()
      img_with_contours = cv2.drawContours(orignal_copy, [i], -1, (0, 255, 0),2)

      area = cv2.contourArea(i)
      peri = cv2.arcLength(i, True)
      approx = cv2.approxPolyDP(i, 0.04 * peri, True)

      if len(approx) >= 6:
        #bbox = cv2.boundingRect(approx)
        areaList.append(area)
        approxList.append(approx)
        #bboxList.append(bbox)
      

    if len(areaList) != 0:
      SortedAreaList= sorted(areaList, reverse=True)
      #print("sorted area list: ",SortedAreaList)
      BiggestIndex = areaList.index(SortedAreaList[0])
      #print(BiggestIndex)
      #print('BiggestIndex: ',BiggestIndex)
      BiggestContour = approxList[BiggestIndex]
      
    print(bboxList)
    return img_with_contours, BiggestContour, BiggestBounding ,cX,cy

  
