#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2021, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: LuisC18

## IMPORTS
import pyrealsense2 as rs
import rospy
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class TOOLBOX_VISION(object):
    """
    Class is a collection of Image Process tools for the support of the tictactoe project.
    """

    def __init__(self):
        self.bridge = CvBridge()


    def camera_info_to_intrinsics(self,cameraInfo):

        intrinsics = rs.intrinsics()

        intrinsics.width = cameraInfo.width
        intrinsics.height = cameraInfo.height
        intrinsics.ppx = cameraInfo.K[2]
        intrinsics.ppy = cameraInfo.K[5]
        intrinsics.fx = cameraInfo.K[0]
        intrinsics.fy = cameraInfo.K[4]
        # intrinsics.model = cameraInfo.distortion_model
        intrinsics.model  = rs.distortion.none     
        intrinsics.coeffs = [i for i in cameraInfo.D]

        return intrinsics
        
    def depth_at_center_pixel(self):

        depth_array = np.array(self.depth_image, dtype=np.float32)
        center_idx = np.array(depth_array.shape) / 2
        center_pixel = (center_idx[0], center_idx[1])
        
        center_pixel_depth = depth_array[center_pixel]

        return center_pixel_depth


    def depth_at_pixel(self,x,y,depth_image):
        

        depth_array = np.array(depth_image, dtype=np.float32)
        pixel = (x, y)

        pixel_depth = depth_array[pixel]

        return pixel_depth

    def convert_depth_to_phys_coord(self,x,y,depth_image,cameraInfo):

        depth = self.depth_at_pixel(x,y,depth_image)
        intrinsics = self.camera_info_to_intrinsics(cameraInfo)

        depth_point_in_meters_camera_cords = rs.rs2_deproject_pixel_to_point(intrinsics, [x,y], depth)
        print(depth_point_in_meters_camera_cords)
        # print("y:",-depth_point_in_meters_camera_cords[0])
        # print("z:",-depth_point_in_meters_camera_cords[1])


def main():
    rospy.init_node('image_processor', anonymous=False)
    TV = TOOLBOX_VISION()
    # IP.depth_at_center_pixel()
    # table estimation  ~ 81 cm, width ~ 60cm
    TV.convert_depth_to_phys_coord(320,100)
    

if __name__ == '__main__':
    main()