#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np
import math
import cv2
import rospy
from std_msgs.msg import *
from os.path import join
import imutils 

print cv2. __version__ #

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Starts streaming
pipeline.start(config)


def shapeContours(frame):
    ### pyimagesearch.com
    #resize image
    #image = cv2.imread('/home/martinez737/ws_pick_camera/src/pick-and-place/Realsense.png')
    image=frame.copy()
    cv2.imshow('test',image)
    rospy.sleep(3)
    
    resized = imutils.resize(image,width = 300)
    ratio = image.shape[0]/float(resized.shape[0])
    #grayscale, blur and threshold
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.threshold(imgBlur,60,255,cv2.THRESH_BINARY)[1]
    # imgCanny = cv2.Canny(imgBlur, 60, 255)
    # kernel = np.ones((3, 3))
    # imgDilate = cv2.dilate(imgCanny, kernel, iterations=3)
    # thresh = cv2.erode(imgDilate, kernel, iterations=2)
    cv2.imshow('thresh',thresh)

    #finds contours and calls shape detector
    contours, _ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    sd = shapeDetect(contours)

    for c in contours:
      print c
      #find center
      M = cv2.moments(c)
      cX = int((M['m10']/(M['m00']+1e-7))*ratio)
      cY = int((M['m01']/(M['m00']+1e-7))*ratio)
    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
      c = c.astype("float")
      c *= ratio
      c = c.astype("int")
      cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
      cv2.putText(image, sd, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
    # show the output image
      cv2.imshow("Image", image)
      cv2.waitKey(0)

  #finds shapes
def shapeDetect(contours):
    ### pyimagesearch.com
    shape = ''
    perimeter = cv2.arcLength(contours,True)

    #approxPolyDP smoothes and approximates the shape of the contour and outputs a set of vertices
    approx = cv2.approxPolyDP(contours,.03 * perimeter, True)

    
    if len(approx)  == 3:
      shape = 'triangle'

    elif len(approx) == 4:
      (x,y,width,height) = cv2.boundingRect(approx)
      aspectRatio = width/float(height)

      shape = 'square' if aspectRatio >= 0.95 and aspectRatio <= 1.05 else 'rectangle'

    elif len(approx) ==5:
      shape = 'pentagon'

    else: 
      shape = 'circle'

    return shape


def main():
    while True:
        try:
        # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
        # Converts images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image_preCrop = np.asanyarray(color_frame.get_data())

            #cropping the color_image to ignore table
            #color_image = color_image_preCrop[60:250, 200:500]
            
            shapeContours(color_image_preCrop)

        except rospy.ROSInterruptException:
            print("ROSInterruptException")
            exit()
        except KeyboardInterrupt:
            exit()


if __name__ == '__main__':
  main()