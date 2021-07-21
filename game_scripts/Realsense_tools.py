#!/usr/bin/env python

import sys                                          # System bindings
import cv2                                          # OpenCV bindings
import numpy as np
import rospy
import pyrealsense2 as rs


#this class is for grabbing frame from the Realsense d435i camera plus image tools

class RealsenseTools:
  def __init__(self):
    self.pipeline = rs.pipeline()
    self.config = rs.config()

    # Get device product line for setting a supporting resolution
    self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
    self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
    self.device = self.pipeline_profile.get_device()
    self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

    self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if self.device_product_line == 'L500':
      self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
      self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Starts streaming
    self.pipeline.start(self.config)
  def grabFrame(self):
    # Starts streaming
    #self.pipeline.start(self.config)
    while True:
      frames = self.pipeline.wait_for_frames()
      color_frame = frames.get_color_frame()
      if not color_frame:
        continue
      else:
        break
    color_image = np.asanyarray(color_frame.get_data())
    # cv2.imshow('test',color_image)
    #print(color_image.shape)
    return color_image


  def cropFrame(self,frame):
    print('Entered RealsenseTools: cropFrame function\n')
    #cropped_image = frame[55:228,335:515] # frame[y,x]
    cropped_image = frame[45:218,315:495 ] # frame[y,x]
    return cropped_image

  def rescaleFrame(self,frame, scale=1):
    print('Entered RealsenseTools: rescaleFrame function')
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

  def deNoise(self,frame): 
    print('Entered RealsenseTools: deNoise function')
  # reduces noise of an image
    image_denoised = cv2.fastNlMeansDenoising(frame,None,40,7,21)
    #cv2.imshow('Denoised Image',image_denoised)
    ''' 
    parameters: fastNlMeansDenoising(
    inputArray: Input 8-bit 1-channel, 2-channel, 3-channel or 4-channel image. 
    OutputArray: Output image with the same size and type as src
    float h: Parameter regulating filter strength. 
            Big h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise
    int template window:(kernel) Size in pixels of the template patch that is used to compute weights.
            Should be odd. Recommended value 7 pixels 
    int SerachWindowsize: Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. 
            Recommended value 21 pixels 
     )
    '''
    return image_denoised