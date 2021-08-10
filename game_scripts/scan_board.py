#!/usr/bin/env python

import numpy as np
import math
import rospy
# from std_msgs.msg import *
import geometry_msgs.msg
import cv2


# import imutils
# include <opencv2/core.hpp> from docs on PCACompute https://docs.opencv.org/4.2.0/d2/de8/group__core__array.html#ga0ad1147fbcdb256f2e14ae2bfb8c991d

class ShapeDetector(object):
    """
    Class that is a collection of shape/object detection tools.
    """
    def __init__(self):
        super(ShapeDetector, self).__init__()

    def detectCircle(self, frame, radius=10, tolerance=10):
        """
        Function that finds and draws circles on an image.
        @param frame: image; Image that will be scanned for circles and drawn on
        @param radius: int; Desired radius of circles found in pixels (default 10 pixels)
        @param tolerance: int; Tolerance for radius size (default 10 pixels)
        @return centerList: List of circle centers in (x ,y) pixels
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)

        rows = blur.shape[0]
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=100, param2=30,
                                   minRadius=radius-tolerance, maxRadius=radius+tolerance)
        centerList = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])  # assuming (i[0]=x, i[1]=y)
                # circle center
                cv2.circle(frame, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(frame, center, radius, (255, 0, 255), 3)
                centerList.append(center)

        # cv2.imshow("detected circles", frame)
        # cv2.waitKey(0)
        # print("Center List:",centerList)
        # circles outputs (x,y, radius)

        return centerList

    def reorder(self, points):
        """
        Function used to reorder points of a detect shape or contour.
        Current application is to consistently reorder bounding box values.
        @param points: List of (x, y) points, corners of shape.
        @return: List of (x, y) reordered points. Order for a square -> (top left, top right, bottom left, bottom right)
        """
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
    def drawAxis(self, img, p_, q_, color, scale):
        # print('Entered Rectangle_support: drawAxis function')
        p = list(p_)
        q = list(q_)

        angle = math.atan2(p[1] - q[1], p[0] - q[0])  # in radians
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
    def getOrientation(self, pts, img):
        """
        OpenCV PCA function to find orientation of a contour.
        https://docs.opencv.org/master/d1/dee/tutorial_introduction_to_pca.html
        @param pts: contour input from opencv's findContours function output
        @param img: Image that the orientation axes will be drawn on
        @return: Z-axis angle of the object
        """
        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i, 0] = pts[i, 0, 0]
            data_pts[i, 1] = pts[i, 0, 1]

        # Performs PCA analysis
        mean = np.empty((0))
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
        # eigenvalues, eigenvectors = np.linalg.eig(mean)

        # Stores the center of the object
        cntr = (int(mean[0, 0]), int(mean[0, 1]))
        # Draws the principal components
        cv2.circle(img, cntr, 3, (255, 0, 255), 2)
        p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
              cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
        p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
              cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
        self.drawAxis(img, cntr, p1, (255, 255, 0), 1)
        self.drawAxis(img, cntr, p2, (255, 80, 255), 1)
        angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
        return angle

        # finds contours on image

    def newOrientation(self, pts):
        """
        Custom orientation function for tictactoe. Takes bounding box reordered approx output points.
        @param pts: reordered points from the boundingRect opencv function.
        @return: Z-axis angle of the board
        """
        # print('shape of points:',pts.shape)
        topleft = [pts[0][0][0], pts[0][0][1]]
        # print('topleft',topleft)
        topright = [pts[1][0][0], pts[1][0][1]]
        # print('topright',topright)
        # print('topright y',topright[1])
        # print('topright x',topright[0])
        slope1_2 = (float(topleft[1]) - float(topright[1])) / (float(topleft[0]) - float(topright[0]))
        # print('slope:',slope1_2)
        angle = math.degrees(math.atan(slope1_2))
        # print('new slope angle',angle)

        return angle

    def getContours(self, orignal_frame):
        """
        OpenCV based function that finds contours on an image
        @param orignal_frame: Image that will be analyzed for contours
        @return: contours output from cv2 findContours function
        """
        self.drawnImage = orignal_frame.copy()
        img_gray = cv2.cvtColor(self.drawnImage, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(img_gray, (7, 7), 0)
        imgCanny = cv2.Canny(imgBlur, 100, 200)

        kernel = np.ones((3, 3))
        imgDilate = cv2.dilate(imgCanny, kernel, iterations=1)
        imgThre = cv2.erode(imgDilate, kernel, iterations=1)
        # cv2.imshow("threshold",imgThre)

        contours, _ = cv2.findContours(imgThre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # contours_image = cv2.drawContours(orignal_frame.copy(), contours, -1, (0, 255, 0), 1)
        # cv2.imshow('all contours seen',contours_image)

        return contours

    def detectSquare(self, contours, area):
        """

        """

        boardCenter = [640, 360]
        boardPoints = [0, 0, 0, 0]
        boardImage = self.drawnImage

        for c in contours:
            # find center using moments
            M = cv2.moments(c)
            cX = int((M['m10'] / (M['m00'] + 1e-7)))
            cY = int((M['m01'] / (M['m00'] + 1e-7)))

            # print('Contours: ',contours)
            perimeter = cv2.arcLength(c, True)

            # approxPolyDP smoothes and approximates the shape of the contour and outputs a set of vertices
            approx = cv2.approxPolyDP(c, .03 * perimeter, True)

            if len(approx) == 4:
                (x, y, width, height) = cv2.boundingRect(approx)
                aspectRatio = float(width) / float(height)
                # print(x,y,width,height)

                if aspectRatio >= 0.95 and aspectRatio <= 1.05 and 350 > height > 200 and 350 > width > 200:  # 640 x480 -> 280 > height > 100 and 280>width>100
                    boardImage = cv2.drawContours(boardImage, [c], -1, (0, 0, 255), 2)
                    # cv2.imshow('board',boardImage)

                    boardCenter[0] = cX
                    boardCenter[1] = cY
                    boardPoints = approx
                    # print(approx)

        return boardImage, boardCenter, boardPoints


    def detectRectangle(self, contours):
