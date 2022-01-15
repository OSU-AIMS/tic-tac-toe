#!/usr/bin/env python

#####################################################
##       Single D435i Frame Capture Tool           ##
##                                                 ##
##   * This is not a ROS enabled node.             ##
##   * Designed to work with a USB connection.     ##
##   * Employs RealSense SDK 2.0                   ##
##                                                 ##
#####################################################
# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Buynak

#####################################################


#####################################################
# Used for Advanced Mode Setup & Point Cloud Streaming
import pyrealsense2 as rs
import time
import sys

# Used for Depth Filtering
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures

# Color Image processing
import cv2

# Maintenance
import logging

#####################################################
#Countdown Timer Allowing for Slow Data Transfer/Collection of one Frame

def timer_wait():
  try:
    for remaining in range(6,0,-1):
      sys.stdout.write("\r")
      sys.stdout.write("Updating Frames: {:2d} seconds remaining.".format(remaining))
      sys.stdout.flush()
      time.sleep(1)
    sys.stdout.write("\r|                                                    \n")
  except KeyboardInterrupt:
    sys.exit()
#####################################################


#####################################################
##    CLASS LEVEL TOOL

class REALSENSE_VISION(object) :

  def intrinsics(self):
    #Camera Intrinsics
    intrProf = self.pipe.get_active_profile()


    self.depth_stream_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
    self.color_stream_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))

    self.intrin_depth = self.depth_stream_profile.get_intrinsics()
    self.intrin_color = self.color_stream_profile.get_intrinsics()

    if False:
      print('Intrin for Depth Stream')
      print(self.intrin_depth)

      print('Intrinsics for Color Stream')
      print(self.intrin_color)

  def enableAdvancedMode(self) :
    """Enables Advanced Mode on Qualifying Realsense Devices"""

    DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6", "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A", "0B5C"]

    def find_device_that_supports_advanced_mode() :
      ctx = rs.context()
      ds5_dev = rs.device()
      devices = ctx.query_devices();
      for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
          if dev.supports(rs.camera_info.name):
            print("Found device that supports advanced mode:", dev.get_info(rs.camera_info.name))
            return dev
      raise Exception("No device that supports advanced mode was found")

    try:
      dev = find_device_that_supports_advanced_mode()
      advnc_mode = rs.rs400_advanced_mode(dev)
      print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

      # Loop until we successfully enable advanced mode
      while not advnc_mode.is_enabled():
        print("Trying to enable advanced mode...")
        advnc_mode.toggle_advanced_mode(True)
        # At this point the device will disconnect and re-connect.
        print("Sleeping for 5 seconds...")
        time.sleep(5)
        # The 'dev' object will become invalid and we need to initialize it again
        dev = find_device_that_supports_advanced_mode()
        advnc_mode = rs.rs400_advanced_mode(dev)
        print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

    # Get each control's current value
    #print("Depth Control: \n", advnc_mode.get_depth_control())
    #print("RSM: \n", advnc_mode.get_rsm())
    #print("RAU Support Vector Control: \n", advnc_mode.get_rau_support_vector_control())
    #print("Color Control: \n", advnc_mode.get_color_control())
    #print("RAU Thresholds Control: \n", advnc_mode.get_rau_thresholds_control())
    #print("SLO Color Thresholds Control: \n", advnc_mode.get_slo_color_thresholds_control())
    #print("SLO Penalty Control: \n", advnc_mode.get_slo_penalty_control())
    #print("HDAD: \n", advnc_mode.get_hdad())
    #print("Color Correction: \n", advnc_mode.get_color_correction())
    #print("Depth Table: \n", advnc_mode.get_depth_table())
    #print("Auto Exposure Control: \n", advnc_mode.get_ae_control())
    #print("Census: \n", advnc_mode.get_census())

    except Exception as e:
      print(e)
      pass

  def setupRSPipeline(self, set_color, set_depth):
    #####################################################
    ##    Setup & Configure Pipeline

    # Declare pointcloud object, for calculating pointclouds and texture mappings
    self.pc = rs.pointcloud()

    # We want the points object to be persistent so we can display the last cloud when a frame drops
    self.points = rs.points()

    # Declare RealSense pipeline, encapsulating the actual device and sensors
    self.pipe   = rs.pipeline()
    config = rs.config()
    opt    = rs.option

    # Set Depth Unit
    opt.depth_units = 0.001  #0.001 for milimeters   #1.0 for meters

    # Enable Depth & Color Streams
    config.enable_stream(rs.stream.depth, set_color[0], set_color[1], rs.format.z16, set_color[2])    #640,480,6   #480,270,30
    config.enable_stream(rs.stream.color, set_depth[0], set_depth[1], rs.format.rgb8, set_depth[2])   #640,480,30   #424,240,30

    # Start streaming with chosen configuration.
    self.profile = self.pipe.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    self.depth_sensor = self.profile.get_device().first_depth_sensor()
    self.depth_scale = self.depth_sensor.get_depth_scale()
    print("Depth Scale is: " , self.depth_scale)

    # We'll use the colorizer to generate texture for our PLY
    # (alternatively, texture can be obtained from color or infrared stream)
    ##self.colorizer = rs.colorizer()

    # Remove Background of Frames greater than clipping_distance_in_meters meters away
    # 2D - RGB image
    clipping_distance_in_meters = self.max_dist #meter
    self.clipping_distance = clipping_distance_in_meters / self.depth_scale
    # 3D - depth image
    self.threshold_filter = rs.threshold_filter()
    self.threshold_filter.set_option(opt.max_distance, self.max_dist)


    # Define Filters
    self.deci_filter = rs.decimation_filter()
    self.deci_filter.set_option(opt.filter_magnitude, 2) #default value (2)

    self.spat_filter = rs.spatial_filter()
    #default values (2,0.5,20,0)
    self.spat_filter.set_option(opt.filter_magnitude,    2)
    self.spat_filter.set_option(opt.filter_smooth_alpha, 0.8)
    self.spat_filter.set_option(opt.filter_smooth_delta, 4)
    self.spat_filter.set_option(opt.holes_fill,          0)

    self.temporal_filter = rs.temporal_filter()
    self.temporal_filter.set_option(opt.filter_smooth_alpha, 0.5)
    self.temporal_filter.set_option(opt.filter_smooth_delta, 20)

    # Create an alignment object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    self.align = rs.align(align_to)

  def __init__(self, set_color=[640,480,30], set_depth=[640,480,30], max_distance=10.0):
    """
    Establish first available Intel Realsense Camera found on USB ports.
    Multiple capture tools are also contained within class once initialized.

    :param set_color: Set Color Stream Resolution (px) and Frequency (Hz)
    :param set_depth: Set Depth Stream Resolution (px) and Frequency (Hz)
    :param max_dist:  Set Max Clipping Distance (max distance from camera)
    """

    self.max_dist = max_distance

    self.enableAdvancedMode()
    self.setupRSPipeline(set_color, set_depth)

    self.intrinsics()

    print("<< Realsense Pipeline Setup \n")
    # timer_wait()



  def capture_singleFrame_alignedRGBD(self,name):
    """
    Captures and Saves Single Aligned RGB-D Frame
    """

    import numpy as np
    from PIL import Image

    try:
      # Wait for the next set of frames from the camera
      frames = self.pipe.wait_for_frames()          #Default wait is 5000 ms

      # Apply Filters to Point Cloud Frames
      frames_filtered = self.applyFilters(frames)

      # Align the depth frame to color frame
      frames_aligned = self.align.process(frames_filtered)

      # Get aligned frames
      depth_frame_aligned = frames_aligned.get_depth_frame()
      color_frame_aligned = frames_aligned.get_color_frame()

      # Validate that both frames are valid
      if not depth_frame_aligned or not color_frame_aligned:
        print("Error! Not Frames not aligned!")

      # Generate Image as Numpy Array
      depth_image = np.asanyarray(depth_frame_aligned.get_data())
      color_image = np.asanyarray(color_frame_aligned.get_data())

      # Remove Background - Set pixels further than clipping_distance to grey
      color_outOfRange = 0   #black = 0   #grey = 153
      depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
      bg_removed = np.where((depth_image_3d > self.clipping_distance) | (depth_image_3d <= 0), color_outOfRange, color_image)

      # Render Image from Color Numpy Array
      image = Image.fromarray(bg_removed)
      save_path_rgb_bg_removed = str(name) + "_rgb_removedBackground.png"
      image.save(save_path_rgb_bg_removed)

      # Render Image from Color Numpy Array (background NOT REMOVED)
      image = Image.fromarray(color_image)


      # save_path_rgb = str(name) + "_rgb.png"
      # image.save(save_path_rgb)

      # # Save Depth Array
      # save_path_depth_npy = str(name) + "_depth.npy"
      # np.save(save_path_depth_npy, depth_image)

    finally:
      print("Image & Depth Array Saved as: " + str(name) )

    return save_path_rgb, save_path_depth_npy

  def capture_singleFrame_depth(self,name):
    """
    Capture Single Frame of Depth Point Cloud and Output as PLY
    Pull frames from Pipeline, Filter, Post-Process
    """

    try:
      # Wait for the next set of frames from the camera
      frames = self.pipe.wait_for_frames()          #Default wait is 5000 ms
      depth_frame = frames.get_depth_frame()

      # Apply Filters
      frames_filtered = self.applyFilters(frames)

      # Apply Colorizer (artificial coloring for depth field)
      colorized = self.colorizer.process(frames_filtered)

      # Export to PLY
      self.export_ply_colorized(colorized, name)

      # Export Depth Frame as NPY
      self.export_npy(frames_filtered, name)

    finally:
      logging.info(">> Completed: Capture_SingleFrame_Depth")

    return 1

  def capture_singleFrame_color(self):
    """
    Capture Static 2D Color Image and Output as PNG
    Pull frames from Pipeline, Filter, Post-Process

    :return True when complete
    """

    import numpy as np
    from PIL import Image

    try:
      # Wait for the next set of frames from the camera
      frames = self.pipe.wait_for_frames()          #Default wait is 5000 ms

      # Get aligned frames
      color_frame = frames.get_color_frame()

      # Generate Image as Numpy Array
      color_image = np.asanyarray(color_frame.get_data())

      # Render Image from Color Numpy Array
      image = Image.fromarray(color_image)
    #   save_path = str(name) + ".png"
    #   image.save(save_path)

    finally:
      print("Static Image outputted:")

    return color_image

  def applyFilters(self,frame):
    """
    Applys filters to provided frame

    Note: Alignment should be performed AFTER post-processing filters are applied.
    This helps reduce distortion effects such as aliasing

    .as_frameset() allows the frames to be filtered as a group and used for later processing.
    """
    frame_filtered = frame
    #frame_filtered = self.deci_filter.process(frame_filtered).as_frameset()
    frame_filtered = self.threshold_filter.process(frame_filtered).as_frameset()
    frame_filtered = self.spat_filter.process(frame_filtered).as_frameset()

    return frame_filtered


  def export_ply_colorized(self, data, save_path):
    # Create PLY Output Instance
    ply = rs.save_to_ply( str(save_path) + ".ply" )

    # Configure Output PLY
    ply.set_option(rs.save_to_ply.option_ply_binary, False)
    ply.set_option(rs.save_to_ply.option_ply_normals, True)

    # Generate Colors for Depth ply
    # Apply Colorizer (artificial coloring for depth field)
    colorized = self.colorizer.process(data)

    # Apply the processing block to the frameset which contains the depth frame and the texture
    ply.process(colorized)

    logging.info(">> PLY saved to " + str(save_path) + ".ply" )

  def export_png(self, data, save_path):
    from PIL import Image
    image = Image.fromarray(data)
    image.save(str(save_path) + ".png")
    logging.info(">> PNG saved to " + str(save_path) + ".png" )

  def export_npy(self, data, save_path):
    with open(str(save_path) + ".npy", 'w') as f:
      np.save(f, data)
    logging.info(">> Numpy Array saved to " + str(save_path) + ".npy" )




  def get_depth_at_point(self, x, y):
    """
    Input X,Y PIXEL coordinates within current camera frame.
    Returns the DEPTH at that pixel location in METERS.
    """

    dpt_frame = self.pipe.wait_for_frames().get_depth_frame().as_depth_frame()
    pixel_distance_in_meters = dpt_frame.get_distance(x,y)

    return pixel_distance_in_meters

  def get_3d_coordinate_at_point(self, depth_pixel_coordinate, depth_value):
    """
    :param depth_intrin:            Camera intrinsics for depth stream
    :param depth_pixel_coordinate:  Desired pixel coordinate in list form [X,Y]
    :param depth_value:             Depth Value in Meters
    :return:                        List of XYZ points
    """

    self.intrinsics()

    depth_intrin = self.intrin_depth

    depth_point_in_meters_camera_coords = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel_coordinate, depth_value)

    return depth_point_in_meters_camera_coords


  def stopRSpipeline(self) :
    # Stop Streaming
    self.pipe.stop()

  def cropFrame(self,frame):
    print('Entered RealsenseTools: cropFrame function\n')
    #cropped_image = frame[55:228,335:515] # frame[y,x]
    cropped_image = frame[45:218,315:495 ] # frame[y,x]
    return cropped_image
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

def main():
  camera = REALSENSE_VISION()
  timer_wait()

  camera.intrinsics()

  #result2 = camera.capture_singleFrame_depth("test2")
  #result3 = camera.capture_singleFrame_color("test3")


  save_path_rgb, save_path_depth_npy = camera.capture_singleFrame_alignedRGBD("test1")

  pixel_x = 480
  pixel_y = 640

  X = (pixel_x - camera_intrinsics.ppx)/camera_intrinsics.fx *depth
  Y = (pixel_y - camera_intrinsics.ppy)/camera_intrinsics.fy *depth










  camera.stopRSpipeline()

if __name__ == '__main__':
  main()
