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
		# cv2.drawContours(cv_image, contours, -1, (0,255,0), 3)
		# cv2.imshow("test",cv_image)
		# cv2.waitKey(0)
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
	quad_area = 0
	# filter quadrilaterals
	if len(approx) == 4:

		# Find area of quadrilateral
		quad_area = cv2.contourArea(contour)

	return quad_area, approx

def orderPoints(points):
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

class TOOLBOX_SHAPE_DETECTOR(object):
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
		tolerance = float(tolerance)/100
		min_area = (1-tolerance) * area 
		max_area = (1+tolerance) * area
	
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

		# # grayscale and blur image
		# gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
		# blur = cv2.medianBlur(gray, 5) # originally 5

		# rows = blur.shape[0] # 720 rows
		# # HoughCircles outputs (x,y, radius)
		# circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, rows / 16,
		# 						   param1=30, param2=25,
		# 						   minRadius=radius - tolerance, maxRadius=radius + tolerance)
		# # original: param1 = 100, param2 = 30

		image_ori = cv2.cvtColor(cv_image,cv2.COLOR_BGR2GRAY)
		lower_bound = np.array([0,0,0])
		
		center_list = []
		upper_bound = np.array([190,220,255])

		image_color = cv_image
		image = image_color

		mask = cv2.inRange(image_color, lower_bound, upper_bound)
		mask = cv2.bitwise_not(mask)
		# cv2.imshow("gray",mask)

		# mask = cv2.adaptiveThreshold(image_ori,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
		#             cv2.THRESH_BINARY_INV,33,2)

		kernel = np.ones((3, 3), np.uint8)

		#Use erosion and dilation combination to eliminate false positives. 
		mask = cv2.erode(mask, kernel, iterations=2)
		mask = cv2.dilate(mask, kernel, iterations=1)

		closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		# cv2.imshow("end",closing)
		# cv2.waitKey(0)


		contours = cv2.findContours(mask.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
		contours.sort(key=lambda x:cv2.boundingRect(x)[0])

		circles = []
		ii = 1
		for c in contours:
		    (x,y),r = cv2.minEnclosingCircle(c)
		    center = (int(x),int(y))
		    r = int(r)
		    if r >= 8 and r<=10:
		        cv2.circle(circles_image,center,r,(61,15,205),3)
		        circles.append(center)

		if circles is not None:
			circles = np.uint16(np.around(circles))
			print circles

			# print('circles[0,:]',circles[0])


			for i in circles:
				
				# circle center
				center = (i[0], i[1])
				
				center_list.append(center)
				# print('i',i)

				# print('center_list',center_list)

		return center_list, circles_image

	def findAngle(self, pts):
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


def main():
	rospy.init_node('image_processor', anonymous=False)
	TV = TOOLBOX_VISION()
	# IP.depth_at_center_pixel()
	# table estimation  ~ 81 cm, width ~ 60cm
	TV.convert_depth_to_phys_coord(320,100)
	

if __name__ == '__main__':
	main()