#!/usr/bin/env python

import sys                                          # System bindings
import cv2                                          # OpenCV bindings
import numpy as np
import rospy
import pyrealsense2 as rs
import time


#this class is for grabbing frame from the Realsense d435i camera plus image tools
def timer_wait(secs=3):
  try:
    for remaining in range(3,0,-1):
      sys.stdout.write("\r")
      sys.stdout.write("Updating Frames: {:2d} seconds remaining.".format(remaining))
      sys.stdout.flush()
      time.sleep(1) 
    sys.stdout.write("\r|                                                    \n")
  except KeyboardInterrupt:
    sys.exit()


class RealsenseTools:
  """
  A class used to initiate python realsense pipeline for the d435i camera.
  Contains tools for grabbing and manipulating images from the pipeline.

  """
  def __init__(self,pipeline=False):
    """Configures the d435i realsense pipeline for depth/color streams: color in 1280x720p and depth in 640x480p"""
    if pipeline:
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
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

      # Starts streaming
      self.pipeline.start(self.config)
      timer_wait()

  def grabFrame(self):
    """
    Grabs a frame from the color frame pipeline.
    @return: A numpy array representation of an rgb color frame from the d435i.
    """
    while True:
      frames = self.pipeline.wait_for_frames()
      color_frame = frames.get_color_frame()
      if not color_frame:
        continue
      else:
        break
    # convert frame to np array
    color_image = np.asanyarray(color_frame.get_data())

    return color_image

  def rescaleFrame(self, frame, scale=1):
    """
    Rescales image to specified scale of original. Default = 1 (no rescale)
    :param frame: The original image that will be rescaled.
    :param scale: The scale that the image will be resized to, 2 = double the size in both width and height.
    :return rescaled_img: Outputs the rescaled image.
    """
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)
    rescaled_img = cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
    return rescaled_img

  def deNoise(self,frame):
    # reduces noise of an image
    image_denoised = cv2.fastNlMeansDenoising(frame,None,40,7,21)
    ''' 
    parameters: fastNlMeansDenoising(
    inputArray: Input 8-bit 1-channel, 2-channel, 3-channel or 4-channel image. 
    OutputArray: Output image with the same size and type as src
    float h: Parameter regulating filter strength. 
            Big h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise
    int template window:(kernel) Size in pixels of the template patch that is used to compute weights.
            Should be odd. Recommended value 7 pixels 
    int SerachWindowsize: Size in pixels of the window that is used to compute weighted average for given pixel. Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. 
            Recommended value 21 pixels)
    '''
    return image_denoised
