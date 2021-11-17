#!/usr/bin/env python

# Script to test smaller kernel convolutions

import cv2

# System Tools
from math import pi, radians, sqrt, atan, ceil
import numpy as np
# import matplotlib.pyplot as plt


def kernel_runner(image):
    # Create kernel (format - "BGR")
    kernel_size = 3
    kernel = np.dstack((255 * np.ones((kernel_size, kernel_size, 1), dtype='uint8'),np.zeros((kernel_size, kernel_size, 3), dtype='uint8'))) # format: BGR
    # numpy dstack docs: https://numpy.org/doc/stable/reference/generated/numpy.dstack.html
    kernel_b = 255 * np.ones((kernel_size, kernel_size), dtype='uint8')


    # Method: Filter2D
    dst = image.copy()
    output = cv2.filter2D(src=image, dst=dst, ddepth=-1, kernel=kernel_b)
    cv2.imshow('Heatmap',output)
    cv2.imshow('Output Array dst', dst)
    cv2.imwrite("dst.tiff", dst)
    cv2.imwrite("result.tff", output)
    '''
    filter2D parameters:
        InputArray src, 
        OutputArray dst, 
        int ddepth, 
        InputArray kernel,
        Point anchor = Point(-1,-1),
        double delta = 0,
        int borderType = BORDER_DEFAULT   
    # opencv docs on filter2D:
    # https://docs.opencv.org/4.2.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    '''


if __name__ == '__main__':
    image = cv2.imread("images_10x10/test_angle.tif")
    # image = image[:, :, 0]
    kernel_runner(image.copy())