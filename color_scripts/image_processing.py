#!/usr/bin/env python

from rectangle_support import *
from color_finder import *
import cv2
#import pyrealsense2 as rs
import rospy
import numpy as np

def main():
  try:
		# Configure depth and color streams
		# pipeline = rs.pipeline()
		# config = rs.config()

		# # Get device product line for setting a supporting resolution
		# pipeline_wrapper = rs.pipeline_wrapper(pipeline)
		# pipeline_profile = config.resolve(pipeline_wrapper)
		# device = pipeline_profile.get_device()
		# device_product_line = str(device.get_info(rs.camera_info.product_line))

		# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

		# if device_product_line == 'L500':
		# 	config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
		# else:
		# 	config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

		# # Starts streaming
		# pipeline.start(config)
		# while True:
		# 	frames = pipeline.wait_for_frames()
		# 	color_frame = frames.get_color_frame()
		# 	if not color_frame:
		# 		continue
		# 	else:
		# 		break

		# color_image = np.asanyarray(color_frame.get_data())
		# cv2.imshow('test',color_image)
		# print(color_image.shape)

		'''
		Need Kernel for better subtraction
		Method below yields black screen

		# Subtract images > Crop > Apply color Filter > Contour Detect
		precrop_image1 = cv2.imread("images/black10_Color.png",cv2.IMREAD_GRAYSCALE)
		#cv2.imshow('Black box-no crop',precrop_image1)
		precrop_image2 = cv2.imread("images/background_Color.png",cv2.IMREAD_GRAYSCALE)
		cv2.imshow('Background-no crop',precrop_image2)
		precrop_image3 = cv2.subtract(precrop_image1,precrop_image2)
		#cv2.imshow('Subtracted image',precrop_image3)
		image3 = precrop_image3[60:250, 200:500]
		cv2.imshow('cropped & subtracted',image3)

		''' 
		'''
		Subtracting images then cropping = cropping then subtracting images

		# image1 = precrop_image1[60:250, 200:500]
		# cv2.imshow('CroppedBlack Box', image1)
		# image2 = precrop_image2[60:250, 200:500]
		# cv2.imshow('Background only', image2)
		# image3 = image1 - image2
		# cv2.imshow('Subtracted image', image3)

		'''
		#image = cv2.imread("images/black10_Color.png",cv2.IMREAD_GRAYSCALE)
		#template = cv2.imread("images/background_Color.png",cv2.IMREAD_GRAYSCALE)
		

############# print(cv2.__version__) OpenCV version: 4.2.0 ################
		print('OpenCV version:',cv2.__version__)
		image = cv2.imread("images/2_Color.png")
		template = cv2.imread("images/back_Color.png")
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
		template = cv2.morphologyEx(template, cv2.MORPH_ERODE, kernel,iterations = 2)
		#template is the background image
		image[template == 0] = 255
		block_precrop = cv2.subtract(template,image)
		block_uninvert = block_precrop[60:250, 200:500]
		cv2.imshow('Block + Background', image)
		cv2.imshow('Background', template)
		cv2.imshow('Should just be block',block_uninvert)
		block = cv2.bitwise_not(block_uninvert)
		cv2.imshow('Inverted Block image',block)
		#cF = colorFilter(block)
		# This displayed the block mostly
		# Picture was large so it included keyboard, table, floor, etc
		# Cropping should fix it

		rospy.sleep(4.5)

		rect = detectRect() # class in rectangle_support.py

		# cF = colorFilter('images/black8_Color.png')
		cF = colorFilter(block)
		grayscale_filtered = cv2.cvtColor(block,cv2.COLOR_BGR2GRAY)
		#grayscale_filtered = image3
		cv2.imshow('Grayscale after cropped & subtracted',grayscale_filtered)
		
		grayscale_filtered = cF.filterBackground()
		cv2.imshow('Grayscale after color filter',grayscale_filtered)
		
## Section below applies Contours
		raw_input('contours <enter>')
		imgContour,bigCont,boundingBox, pixel_x, pixel_y = rect.getContours(cF.original,grayscale_filtered)
		#imgContour,bigCont,boundingBox, pixel_x, pixel_y = rect.getContours(cF.original,binary_image)
		cv2.imshow('Contours',imgContour)

		drawingFit = cF.original.copy()
		# rospy.sleep(5)
		# Locates center point
		nPoints= rect.reorder(bigCont)


		#draw centerpoint
		cv2.circle(drawingFit, (pixel_x,pixel_y), 0, (0,0,255), 3)
		print('x center pixel:',pixel_x)
		print('y center pixel:',pixel_y)

		#need pixel to cm conversion
		scale = 66.6/640 #cm/pixel
		centimeter_x = pixel_x * scale
		centimeter_y = pixel_y *scale

		angle, cntr, mean = rect.getOrientation(nPoints, imgContour)
		result_angle = int(np.rad2deg(angle))

		cv2.imshow('Fit',drawingFit)

		print('********* RESULTS ***************')
		print('Angle is ' + str(result_angle) + ' degrees [CW Positive]')
		print('Coordinate of center is (' + str(centimeter_x) + ' , ' + str(centimeter_y) + ') cm')

		cv2.waitKey(0)
		cv2.destroyAllWindows()

  except rospy.ROSInterruptException:
    exit()
  except KeyboardInterrupt:
    exit()

if __name__ == '__main__':
  main()
