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
    kernel = np.dstack((255 * np.ones((kernel_size, kernel_size, 1), dtype='uint8'),np.zeros((kernel_size, kernel_size, 2), dtype='uint8')))
    # 11/22: changed dtype from uinut8 to np.float32 --> didn't change anything
    # Changed np.zeros(kernel_size,kernel_size,3)  to np.zeros(kernel_size,kernel_size,2) so kernel is 3 channels instead of 4
    # format: BGR
    # numpy dstack docs: https://numpy.org/doc/stable/reference/generated/numpy.dstack.html

    # Uncomment below to use white kernel 3x3x3
    # kernel_b = 255 * np.ones((kernel_size, kernel_size), dtype='uint8')

    # Attempting to Create a blue (255,0,0) 3x3x3 Kernel
    # kernel_b = np.dstack((255 * np.ones((kernel_size, kernel_size,1), dtype='uint8'), np.zeros((kernel_size, kernel_size, 2), dtype='uint8')))
    kernel_b = cv2.imread('tic_tac_toe_images/blue_square_crop.tiff')

    # print('Kernel Matrix: 3x3x1')
    # print(kernel_b)


    # Method: Filter2D
# <<<<<<< Updated upstream
    # dst = image.copy()
    # output = cv2.filter2D(src=image, dst=dst, ddepth=-1, kernel=kernel_b)
    # cv2.imshow('Input Array',image)
    # cv2.imshow('Heatmap',output)
    # cv2.imshow('Output Array dst', dst)
    # cv2.imwrite("dst.tiff", dst)
    # cv2.imwrite("output.tiff", output)
    # wouldn't output = dst??
    # 11/18: Result is not a gradient
    # kernel_b=cv2.flip(kernel_b,-1)
    # dst = image.copy()
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # output = cv2.filter2D(src=image, ddepth=-1, kernel=kernel_b)
    # 11/22: removed dst=dst
    # cv2.imshow('Heatmap',output)
    # cv2.imshow('Output Array dst', dst)
    # cv2.imwrite('dst.tiff', dst) dst & output are the same
    # cv2.imwrite('output_filter2D.tif', output)

    '''
    filter2D parameters:
        InputArray src, 
        OutputArray dst, of same size & same number of channels as src
        int ddepth, desired depth of destination img
        InputArray kernel, 
        Point anchor = Point(-1,-1),
        double delta = 0,
        int borderType = BORDER_DEFAULT   
    # opencv docs on filter2D:
    # https://docs.opencv.org/4.2.0/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    '''

 # Uncomment below after testing out filter2D function & get a consistent output

    # # # Method: Template Matching
    # # image = image[:, :, 0]
    # img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # # resb = cv2.matchTemplate(image=image[:, :, 0], templ=kernel_b, method=1)
    # # resg = cv2.matchTemplate(image=image[:, :, 1], templ=kernel_b, method=1)
    # # resr = cv2.matchTemplate(image=image[:, :, 2], templ=kernel_b, method=1)
    # # cv2.imwrite('resb.tiff',resb)
    # # cv2.imwrite('resg.tiff',resg)
    # # cv2.imwrite('resr.tiff',resr)
    # # template = kernel_b[1,:,:]
    '''
    matchTemplate docs
    https://docs.opencv.org/4.2.0/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dac6677e2af5e0fae82cc5339bfaef5038
        Input Array: image (must be 8 bit or 32 bit floating point)
        Input array: Templ (serached template)
       output array: result
               int: method (https://docs.opencv.org/4.2.0/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d)
               mask: mask of serached template. Same datatype & size as templ. Not set by default
    '''


    res = cv2.matchTemplate(image=image,templ=kernel_b,method=5)
    cv2.imwrite('res_match_template.tiff',res)
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print('min_val')
    print(min_val)
    print('max_val')
    print(max_val)
    print('min_loc')
    print(min_loc)
    print('max_loc')
    # print(max_loc)

    # Drawing Bounding Box around detected shape
    # determine the starting and ending (x, y)-coordinates of the bounding box
    # From: https://www.pyimagesearch.com/2021/03/22/opencv-template-matching-cv2-matchtemplate/
    (startX, startY) = max_loc
    endX = startX + kernel_b.shape[1]
    endY = startY + kernel_b.shape[0]
    
    # draw the bounding box on the image
    b_box_image = cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 1)
    # show the output image
    # cv2.imshow("Output based on matchTemplate", b_box_image)
    cv2.imwrite('res_match_template_BoundingBox.tiff',b_box_image)
    cv2.waitKey(0)

'''
    cv::TemplateMatchModes 
    cv::TM_SQDIFF = 0,
    cv::TM_SQDIFF_NORMED = 1,
    cv::TM_CCORR = 2,
    cv::TM_CCORR_NORMED = 3,
    cv::TM_CCOEFF = 4,
    cv::TM_CCOEFF_NORMED = 5
'''

    # # Filter Results ##############
    # result = (resb*255) - (resg*255) - (resr*255)
    # flag_negatives = result < 0
    # result[flag_negatives] = 0
    # cv2.imshow('Result', result/255)
    # cv2.imwrite("res.tiff", result)


if __name__ == '__main__':
    print("Your OpenCV version is: " + cv2.__version__)  
    # image = cv2.imread("images_10x10/diamond.tif")
    image = cv2.imread("tic_tac_toe_images/twistCorrectedColoredSquares_Color.tiff")

    # image = cv2.imread("images_20x20/test_box_20x20.tif")
    # image = image[:, :, 0]
    kernel_runner(image.copy())