#!/usr/bin/env python

import numpy as np
import math
import cv2


def reorder(points):
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


def findDis(pts1, pts2):
    """
    Finds the distance between 2 points (distance formula)
    @param pts1: First point (x, y)
    @param pts2: Second point (x, y)
    @return: Distance between 2 points
    """

    x1 = float(pts1[0])
    x2 = float(pts2[0])
    y1 = float(pts1[1])
    y2 = float(pts2[1])
    dis = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    return dis


class ShapeDetector(object):
    """
    Class is a collection of shape/object detection tools.
    """

    def __init__(self):
        super(ShapeDetector, self).__init__()
        self.drawnImage = None

    def detectSquare(self, image, area):
        """
        Custom Function to find and display a single square of a specific size in an image.
        @param image: Input Image that the square will be drawn on.
        @param area: int area of desired square
        @return ImageWithSquare: Image with drawn square for visual purposes.
        @return SquareCenter: (x, y) center pixel location of square on image.
        @return SquarePoints: (x, y) pixel points of the square (top left, top right, bot left, bot right)
        """

        if image is None:
            print(">> detectSquare: No image to draw rectangle on!")
            exit()
        else:
            ImageWithSquare = image

        SquareCenter = [640, 360]
        SquarePoints = [0, 0, 0, 0]
        # TODO: ^^^ fix use in board_center_publisher.py ^^^
        # find length of side and diagonal
        side = math.sqrt(area)
        diag = side * math.sqrt(2)
        foundSquare = False

        # find all contours and filter
        contours = self.getContours(ImageWithSquare)
        for c in contours:

            # find center using moments
            M = cv2.moments(c)
            cX = int((M['m10'] / (M['m00'] + 1e-7)))
            cY = int((M['m01'] / (M['m00'] + 1e-7)))

            # Find perimeter of contour
            perimeter = cv2.arcLength(c, True)

            # approxPolyDP smooths and approximates the shape of the contour and outputs a set of vertices
            approx = cv2.approxPolyDP(c, .03 * perimeter, True)

            # filter quadrilaterals
            if len(approx) == 4:
                # define aspect ratio of quadrilateral
                (x, y, width, height) = cv2.boundingRect(approx)
                # height/width is measured by farthest x points and y points; hence the use of the sqrt(2) diagonal
                aspectRatio = float(width) / float(height)

                # check if square with correct area
                if 0.95 <= aspectRatio <= 1.05 and 1.05 * diag > height > side * .95:
                    ImageWithSquare = cv2.drawContours(ImageWithSquare, [c], -1, (0, 0, 255), 2)

                    SquareCenter[0] = cX
                    SquareCenter[1] = cY
                    SquarePoints = reorder(approx)
                    foundSquare = True
                    break

        if not foundSquare:
            print('No suitable square found!')
            exit()
        elif foundSquare:
            return ImageWithSquare, SquareCenter, SquarePoints

    def detectRectangle(self, image, area, tolerance=10):
        """
        Custom Function to detect a single rectangle of a specific size.
        @param area: int area of desired rectangle (in pixels)
        @param image: Input Image that the square will be found and drawn on
        @param tolerance: percentage of tolerance allowed for finding area (default is 10%)
        @return RectCenter: (x, y) center pixel location of rectangle on image.
        @return ImageWithRect: Image with drawn Rectangle for visual purposes.
        @return RectPoints: (x, y) pixel points of the rectangle (top left, top right, bot left, bot right)
        """

        if image is None:
            print(">> detectRectangle: No image to draw rectangle on!")
            exit()
        else:
            ImageWithRect = image

        # Define Variables
        RectCenter = []
        RectPoints = [0,0,0,0]                      # expecting 4 points
        toleranceScale = tolerance / 100
        foundRect = False

        # Find all contours on image
        contours = self.getContours(ImageWithRect)

        # Filter through found contours
        for c in contours:
            # find center using moments
            M = cv2.moments(c)
            cX = int((M['m10'] / (M['m00'] + 1e-7)))
            cY = int((M['m01'] / (M['m00'] + 1e-7)))

            # Find Perimeter
            perimeter = cv2.arcLength(c, True)

            # approxPolyDP smooths and approximates the shape of the contour and outputs a set of vertices
            approx = cv2.approxPolyDP(c, .03 * perimeter, True)

            # filter quadrilaterals
            if len(approx) == 4:
                # (x, y, width, height) = cv2.boundingRect(approx)

                # Find area of quadrilateral
                contourArea = cv2.contourArea(c)

                # Check if area matches area param
                if (1+toleranceScale) * area > contourArea > (1-toleranceScale) * area:
                    # Draw the Rectangle
                    ImageWithRect = cv2.drawContours(ImageWithRect, [c], -1, (0, 0, 255), 2)

                    # Define center and points
                    RectCenter[0] = cX
                    RectCenter[1] = cY
                    RectPoints = reorder(approx)

                    foundRect = True
                    break
        if not foundRect:
            print('No suitable rectangle found!')
            exit()
        if foundRect:
            return RectCenter, ImageWithRect, RectPoints

    def detectCircle(self, image, radius=10, tolerance=10):
        """
        Function that finds and draws circles on an image.
        @param image: image; Image that will be scanned for circles and drawn on
        @param radius: int; Desired radius of circles found in pixels (default 10 pixels)
        @param tolerance: int; Tolerance for radius size (default 10 pixels)
        @return centerList: List of circle centers in (x ,y) pixels
        """
        # grayscale and blur image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)

        rows = blur.shape[0]
        # circles outputs (x,y, radius)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=100, param2=30,
                                   minRadius=radius - tolerance, maxRadius=radius + tolerance)

        centerList = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])  # assuming (i[0]=x, i[1]=y)
                # circle center
                cv2.circle(image, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(image, center, radius, (255, 0, 255), 3)
                centerList.append(center)

        return centerList

    def getOrientation(self, pts, img):
        """
        OpenCV PCA function to find orientation(rotation angle and center) of a contour.
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

    def newOrientation(self, pts):
        """
        Custom orientation function for tictactoe. Takes bounding box reordered approx output points.
        @param pts: reordered points from the boundingRect opencv function.
        @return: Z-axis angle of the board
        """

        sz = len(pts)
        data_pts = np.empty((sz, 2), dtype=np.float64)
        for i in range(data_pts.shape[0]):
            data_pts[i, 0] = pts[i, 0, 0]
            data_pts[i, 1] = pts[i, 0, 1]

        # define top right and left points
        topleft = [pts[0][0][0], pts[0][0][1]]
        topright = [pts[1][0][0], pts[1][0][1]]

        # use point-slope formula
        slope1_2 = (float(topleft[1]) - float(topright[1])) / (float(topleft[0]) - float(topright[0]))

        # convert to radians
        angle = math.degrees(math.atan(slope1_2))

        return angle

    def getContours(self, original_frame):
        """
        OpenCV based function that finds contours on an image
        @param original_frame: Image that will be analyzed for contours
        @return: contours output from cv2 findContours function
        """
        # Copy original image so it does not get overwritten
        self.drawnImage = original_frame.copy()

        # grayscale image
        img_gray = cv2.cvtColor(self.drawnImage, cv2.COLOR_BGR2GRAY)

        # apply gaussian blur
        imgBlur = cv2.GaussianBlur(img_gray, (7, 7), 0)

        # apply canny
        imgCanny = cv2.Canny(imgBlur, 100, 200)

        # Dilate and Erode Image: results in threshold image
        kernel = np.ones((3, 3))
        imgDilate = cv2.dilate(imgCanny, kernel, iterations=1)
        imgThresh = cv2.erode(imgDilate, kernel, iterations=1)

        # Useful for tuning dilate, erode, kernel or threshold values
        # cv2.imshow("Image threshold", imgThresh)

        # Finds all contours on image (using threshold image)
        contours, _ = cv2.findContours(imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def drawAxis(self, img, p_, q_, color, scale):
        """
        Draws x-y axis relative to object center and orientation, second part of the getOrientation function.
        https://docs.opencv.org/master/d1/dee/tutorial_introduction_to_pca.html
        @param img: Image the orientation axis are drawn on
        @param p_: Center point of contour for axis origin
        @param q_: Principal components points for axis end point
        @param color: rgb 0 - 255 color values for drawn axis
        @param scale: int scale for length of axis
        """
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
