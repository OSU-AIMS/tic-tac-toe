#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np
import math
import cv2
import rospy
from std_msgs.msg import *
import geometry_msgs.msg 
from rectangle_support import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def callback(data):
    try:
      cv_image = CvBridge().imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(1)
    return cv_image


def main():
  try:
    rect = detectRect()
    # Makes list for coords and angle
    index = 0
    xList = []
    yList = []
    angleList = []


   
    rospy.init_node('image_converter', anonymous=True)
    sub_image = rospy.Subscriber("/camera/c",Image,callback)


    #depth_image = np.asanyarray(depth_frame.get_data())
    color_img_preCrop = np.asanyarray(sub_image.get_data())

    #cropping the color_img to ignore table
    color_img = color_img_preCrop[60:250, 200:500]

    cv2.imshow("Cropped color image",color_img)

    # Applys colormap on depth image (image must be converted to 8-bit per pixel first)
    #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    # depth_colormap_dim = depth_colormap.shape
    # color_colormap_dim = color_img.shape

    # # If depth and color resolutions are different, resize color image to match depth image for display
    # if depth_colormap_dim != color_colormap_dim:
    #     resized_color_img = cv2.resize(color_img, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
    #                                      interpolation=cv2.INTER_AREA)
    #     images = np.hstack((resized_color_img, depth_colormap))
    # else:
    #     images = np.hstack((color_img, depth_colormap))



    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    #cv2.imshow('RealSense',images)

    image = color_img.copy()

    #imgDepth=depth_frame.copy()
    #cv2.imshow('Depth',depth_frame)
    #cv2.imshow('Depth color',depth_colormap)
    #print('Depth Array',depth_frame)

    cv2.imshow('RealSense',image)

    rect.shapeContours(image)



  except rospy.ROSInterruptException:
    exit()
  except KeyboardInterrupt:
    exit()

if __name__ == '__main__':
  main()