#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2021, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: LuisC18

## IMPORTS

import numpy as np
import math
import cv2
from scipy.spatial import distance as dist

def getContours(cv_image):
        """
        OpenCV based function that finds contours on an image
        @param original_frame: Image that will be analyzed for contours
        @return: contours output from cv2 findContours function
        """

        # Copy original image 
        cv_image = cv_image.copy()

        # grayscale image
        img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # apply gaussian blur
        img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)

        # apply canny
        img_canny = cv2.Canny(img_blur, 100, 200)

        # Dilate and Erode Image: results in threshold image
        kernel = np.ones((3, 3))
        img_dilate = cv2.dilate(img_canny, kernel, iterations=1)
        img_thresh = cv2.erode(img_dilate, kernel, iterations=1)

        # Finds all contours on image (using threshold image)
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours

def momentCenter(contour):
	moments = cv2.moments(contour)

    center_x = int((moments['m10'] / (moments['m00'] + 1e-7)))
    center_y = int((moments['m01'] / (moments['m00'] + 1e-7)))

    return [center_x,center_y]

def filterQuadrilaterals(contour):

    perimeter = cv2.arcLength(contour, True)

    # approxPolyDP smooths and approximates the shape of the contour and outputs a set of vertices
    approx = cv2.approxPolyDP(contour, .03 * perimeter, True)

    # filter quadrilaterals
    if len(approx) == 4:

        # Find area of quadrilateral
        quad_area = cv2.contourArea(c)

    return quad_area, approx

def orderPoints(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]

	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]

	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	
	return np.array([tl, tr, bl, br], dtype="float32")

class ShapeDetector(object):
	"""
    Class is a collection of shape detection tools based in opencv Image tools. Function image inputs require opencv image types.
    """
    def __init__(self):
    	pass
    	
    def detectSquare(self, cv_image, area, tolerance = 5):
        """
        Custom Function to find and display a single square of a specific size in an image.
        @param image: Input Image that the square will be drawn on.
        @param area: int area of desired square in pixels
        @return image_with_square: Image with drawn square for visual purposes.
        @return square_center: (x, y) center pixel location of square on image.
        @return square_points: (x, y) pixel points of the square (top left, top right, bottom left, bottom right)
        """
        
        min_area = (1-tolerance/100) * area 
        max_area = (1+tolerance/100) * area 

        contours = getContours(cv_image)

        for c in contours:
            quad_area, quad_points = filterQuadrilaterals(c)

            (x, y, width, height) = cv2.boundingRect(quad_points)
            aspect_ratio = float(width) / float(height)

                # aspect_ratio of 1 is a square
                if 0.95 <= aspect_ratio <= 1.05 and min_area < quad_area < max_area:

                    image_with_square = cv2.drawContours(cv_image, [c], -1, (255, 36, 0), 3)
                    square_points = orderPoints(quad_points)
                    square_center = momentCenter(c)
                    
                    return image_with_square, square_center, square_points


    def detectRectangle(self, cv_image, area, tolerance=5):
        """
        Custom Function to detect a single rectangle of a specific size.
        @param area: int area of desired rectangle (pixels^2)
        @param image: Input Image that the square will be found and drawn on
        @param tolerance: percentage of tolerance allowed for finding area (default is 10%)
        @return image_with_rect: Image with drawn Rectangle for visual purposes.
        @return rect_center: (x, y) center pixel location of rectangle on image.
        @return rect_points: (x, y) pixel points of the rectangle (top left, top right, bottom left, bottom right)
        """

        min_area = (1-tolerance/100) * area 
        max_area = (1+tolerance/100) * area 

        contours = getContours(cv_image)

        for c in contours:
        	quad_area, quad_points = filterQuadrilaterals(c)

                if min_area < quad_area < max_area:

                    image_with_rect = cv2.drawContours(cv_image, [c], -1, (0, 0, 255), 2)

                  	rect_center = momentCenter(c)
                    rect_points = reorder(quad_points)

            		return image_with_rect, rect_center, rect_points


    def detectCircles(self, cv_image, radius=0, tolerance=0):
        """
        Function that finds and draws circles on an image.
        @param image: image; Image that will be scanned for circles and drawn on
        @param radius: int; Desired radius of circles found in pixels (default finds all circles)
        @param tolerance: int; Tolerance for radius size (default is 0)
        @return center_list: List of circle centers in (x ,y) pixels
        """

        circles_image = cv_image.copy()

        # grayscale and blur image
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)

        rows = blur.shape[0]
        # HoughCircles outputs (x,y, radius)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                   param1=100, param2=30,
                                   minRadius=radius - tolerance, maxRadius=radius + tolerance)

        center_list = []

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                
                # circle center
                center = (i[0], i[1])
                cv2.circle(circles_image, center, 1, (0, 100, 100), 3)

                # circle outline
                radius = i[2]
                cv2.circle(circles_image, center, radius, (255, 0, 255), 3)

                center_list.append(center)

        return center_list, circles_image


def main():
    rospy.init_node('image_processor', anonymous=False)
    TV = TOOLBOX_VISION()
    # IP.depth_at_center_pixel()
    # table estimation  ~ 81 cm, width ~ 60cm
    TV.convert_depth_to_phys_coord(320,100)
    

if __name__ == '__main__':
    main()