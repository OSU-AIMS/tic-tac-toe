#!/usr/bin/env python

#import pyrealsense2 as rs
import numpy as np
import math
import rospy
#from std_msgs.msg import *
import geometry_msgs.msg 
import cv2
#import imutils

class detectRect(object):
  def __init__(self):
    super(detectRect,self).__init__()
    #rospy.init_node('node_detectRectangle',anonymous=True)
    from geometry_msgs.msg import Pose
    #self.sendPosOrient= rospy.Publisher('Coordinates/Angle', Pose, queue_size=10)

    # self.pipeline = rs.pipeline()
    # self.config = rs.config()

    # # Get device product line for setting a supporting resolution
    # self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
    # self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
    # self.device = self.pipeline_profile.get_device()
    # self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

    # self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # if self.device_product_line == 'L500':
    #   self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    # else:
    #   self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # # Starts streaming
    # self.pipeline.start(self.config)


  def isSame(self,index, xList, yList, angleList):
    if index>=20:
      j=index-19
    else:
      j = 0
    countX = 0
    countY = 0
    countA = 0
    while (j < (index-1)):
      if (xList[j] >= (xList[j+1] - 1) and xList[j] <= (xList[j+1] + 1)):
        countX = countX + 1
      if (yList[j] >= (yList[j+1] - 1) and yList[j] <= (yList[j+1] + 1)):
        countY = countY + 1
      if (angleList[j] >= (angleList[j+1] - 3) and angleList[j] <= (angleList[j+1] + 3)):
        countA = countA + 1
      j = j + 1
    if (countX == 18 & countY == 18 & countA == 18):
      same = True
      return same
    else:
      same = False
      return same

    # Reorders the four points of the rectangle
  def reorder(self,points):
    try:
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

  # Finds the distance between 2 points (distance formula)
  def findDis(self,pts1, pts2):
    x1 = float(pts1[0])
    x2 = float(pts2[0])
    y1 = float(pts1[1])
    y2 = float(pts2[1])
    dis = ((x2 - x1)**2 + (y2 - y1)**2)**(0.5)
    return dis

  # Draws x-y axis relative to object center and orientation
  def drawAxis(self,img, p_, q_, color, scale):
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
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
      data_pts[i, 0] = pts[i, 0, 0]
      data_pts[i, 1] = pts[i, 0, 1]

    # Performs PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

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

    imgBlur = cv2.GaussianBlur(gray_mask, (7, 7), 0)
    #cv2.imshow('blur',imgBlur)
    imgCanny = cv2.Canny(imgBlur, 100, 200)

    kernel = np.ones((3, 3))
    imgDilate = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDilate, kernel, iterations=2)

    cv2.imshow("img threshold",imgThre)
    contours, _ = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areaList = []
    approxList = []
    bboxList = []
    BiggestContour = []
    BiggestBounding =[]
    for i in contours:

      #find center using moments
      M = cv2.moments(i)
      cX = int((M['m10']/M['m00']))
      cy = int((M['m01']/M['m00']))
      #print('Contours: ',contours)

      orignal_copy = orignal_frame.copy()
      img_with_contours = cv2.drawContours(orignal_copy, [i], -1, (0, 255, 0),2)

      area = cv2.contourArea(i)
      peri = cv2.arcLength(i, True)
      approx = cv2.approxPolyDP(i, 0.04 * peri, True)

      if len(approx) == 4:
        bbox = cv2.boundingRect(approx)
        areaList.append(area)
        approxList.append(approx)
        bboxList.append(bbox)
    if len(areaList) != 0:
      SortedAreaList= sorted(areaList, reverse=True)
      #print("sorted area list: ",SortedAreaList)
      BiggestIndex = areaList.index(SortedAreaList[0])
      #print('BiggestIndex: ',BiggestIndex)
      BiggestContour = approxList[BiggestIndex]
      BiggestBounding = bboxList[BiggestIndex]

    print(bboxList)
    return img_with_contours, BiggestContour, BiggestBounding ,cX,cy

  # def shapeContours(self,frame):
  #   ### pyimagesearch.com
  #   #resize image
  #   image = cv2.imread(frame)
  #   resized = imutils.resize(image,width = 300)
  #   ratio = image.shape[0]/float(resized.shape[0])

  #   #grayscale, blur and threshold
  #   imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #   imgBlur = cv2.GaussianBlur(image, (5, 5), 0)
  #   thresh = cv2.threshold(imgBlur,60,255,cv2.THRESH_BINARY)[1]

  #   #finds contours and calls shape detector
  #   contours = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  #   contours = imutils.grab_contours(contours)
  #   sd = shapeDetect(contours)

  #   for c in contours:
  #     #find center
  #     M = cv2.moments(c)
  #     cX = int((M['m10']/M['m00'])*ratio)
  #     cy = int((M['m01']/M['m00'])*ratio)


  #     c = c.astype("float")
  #     c *= ratio
  #     c = c.astype("int")
  #     cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
  #     cv2.putText(image, sd, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
  #   # show the output image
  #     cv2.imshow("Image", image)
  #     cv2.waitKey(0)

  # #finds shapes
  # def shapeDetect(self,contours):
  #   ### pyimagesearch.com
  #   shape = ''
  #   perimeter = cv2.arcLength(contours,True)

  #   #approxPolyDP smoothes and approximates the shape of the contour and outputs a set of vertices
  #   approx = cv2.approxPolyDP(contours,.03 * perimeter, True)

    
  #   if len(approx)  == 3:
  #     shape = 'triangle'

  #   elif len(approx) == 4:
  #     (x,y,width,height) = cv2.boundingRect(approx)
  #     aspectRatio = width/float(height)

  #     shape = 'square' if aspectRatio >= 0.95 and aspectRatio <= 1.05 else 'rectangle'

  #   elif len(approx) ==5:
  #     shape = 'pentagon'

  #   else:
  #     shape = 'circle'

  #   return shape

  def talker():
    #pub = rospy.Publisher('Coordinates/Angle',Pose, queue_size=10)
    # is float64 right for Python 2.7? 
    from geometry_msgs.msg import Pose
    print('Initialized Node')
    pub = rospy.Publisher('cameraPose',Pose, queue_size=10) #TODO: couldn't get latch=True to work. Looping instead
    rospy.init_node('cameraPose', anonymous=False)
    rate = rospy.Rate(1) # 10hz

    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = xC
    pose_goal.position.y = yC
    pose_goal.orientation.z = pub_angle


    # Publish node
    while not rospy.is_shutdown():
        #rospy.loginfo(message)
        pub.publish(pose_goal)
        rate.sleep()