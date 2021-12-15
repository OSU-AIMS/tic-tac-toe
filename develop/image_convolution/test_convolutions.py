#!/usr/bin/env python

# Script to test smaller kernel convolutions

import cv2

# System Tools
from math import pi, radians, sqrt, atan, ceil
import numpy as np
import matplotlib.pyplot as plt


def kernel_runner(image):
    # Create kernel (format - "BGR")
    kernel_size = 5
    print('Image Parameter')
    # print(image)
    print('Shape of Input image')
    print(np.shape(image))
    # kernel = np.dstack((255 * np.ones((kernel_size, kernel_size, 1), dtype='uint8'),np.zeros((kernel_size, kernel_size, 2), dtype='uint8')))
    # 11/22: changed dtype from uinut8 to np.float32 --> didn't change anything
    # Changed np.zeros(kernel_size,kernel_size,3)  to np.zeros(kernel_size,kernel_size,2) so kernel is 3 channels instead of 4
    # format: BGR
    # numpy dstack docs: https://numpy.org/doc/stable/reference/generated/numpy.dstack.html

    # Uncomment below to use white kernel 3x3x3
    # kernel_b = 255 * np.ones((kernel_size, kernel_size), dtype='uint8')

    # Uncomment below to create blue array
    ch1 = 255*np.ones((kernel_size, kernel_size), dtype='uint8')
    ch2 = np.zeros((kernel_size, kernel_size), dtype='uint8')
    kernel_b = np.array([ch1, ch2, ch2], ndmin=3, dtype='uint8')
    # might be BGR

    # kernel_b = np.stack((ch2,ch2,ch1),axis=-1)
    # 12/6: issue with creating 3 channel array
    # need (0,0,255) 3 channel array
    # Currently makes [0,0,255],[0,0,255]. Need [0,0,0],[0,0,0],[255,255,255]


    # Attempting to Create a blue (255,0,0) 3x3x3 Kernel
    # kernel_b = np.dstack((255 * np.ones((kernel_size, kernel_size,1), dtype='uint8'), np.zeros((kernel_size, kernel_size, 2), dtype='uint8')))



    # Uncomment below to use square images as kernels
    # kernel_b = cv2.imread('tic_tac_toe_images/blue_square_crop.tiff')

    # kernel_r = cv2.imread('tic_tac_toe_images/red_square_crop.tiff')

    # kernel_g = cv2.imread('tic_tac_toe_images/green_square_crop.tiff')

    print('Kernel Matrix: should be 3x3x3')
    print(np.shape(kernel_b)) # returns 3x3x3
    print(kernel_b)


##### Method: Filter2D
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

    # # Recognizing Blue Square
    print('Using matchTemplate() function')
    res_B = cv2.matchTemplate(image=image,templ=kernel_b,method=5)
    # Use method=5 when using the square images as kernels
    # Use method= when using arrays as kernels
    cv2.imwrite('res_match_template_B.tiff', res_B)
    min_val_B, max_val_B, min_loc_B, max_loc_B = cv2.minMaxLoc(res_B)
    # print('min_val_B')
    # print(min_val_B)
    # print('max_val_B')
    # print(max_val_B)
    print('min_loc_B')
    print(min_loc_B)
    print('max_loc_B')
    print(max_loc_B)

    # Drawing Bounding Box around detected shape
    # determine the starting and ending (x, y)-coordinates of the bounding box
    # From: https://www.pyimagesearch.com/2021/03/22/opencv-template-matching-cv2-matchtemplate/
    (startX_B, startY_B) = max_loc_B
    endX_B = startX_B + kernel_b.shape[1]
    endY_B = startY_B + kernel_b.shape[0]

    # draw the bounding box on the image
    b_box_image = cv2.rectangle(image, (startX_B, startY_B), (endX_B, endY_B), (255, 0, 0), 4) # BGR for openCV
    # show the output image
    # cv2.imshow("Output based on matchTemplate", b_box_image)
    cv2.imwrite('res_match_template_Blue_BoundingBox.tiff', b_box_image)
    plt.figure(1)
    plt.imshow(b_box_image)
    plt.show()
    # cv2.waitKey(0)

    #### Recognizing Red Square
    # res_R = cv2.matchTemplate(image=image,templ= kernel_r,method=5)
    # cv2.imwrite('res_match_template_R.tiff', res_R)
    # min_val_R, max_val_R, min_loc_R, max_loc_R = cv2.minMaxLoc(res_R)
    # # print('min_val_R')
    # # print(min_val_R)
    # # print('max_val_R')
    # # print(max_val_R)
    # print('min_loc_R')
    # print(min_loc_R)
    # print('max_loc_R')
    # print(max_loc_R)
    #
    # # Drawing Bounding Box around detected shape
    # # determine the starting and ending (x, y)-coordinates of the bounding box
    # # From: https://www.pyimagesearch.com/2021/03/22/opencv-template-matching-cv2-matchtemplate/
    # (startX_R, startY_R) = max_loc_R
    # endX_R = startX_R + kernel_r.shape[1]
    # endY_R = startY_R + kernel_r.shape[0]
    #
    # # draw the bounding box on the image
    # r_box_image = cv2.rectangle(image, (startX_R, startY_R), (endX_R, endY_R), (0, 0, 255), 3)
    # # show the output image
    # # cv2.imshow("Output based on matchTemplate", r_box_image)
    # cv2.imwrite('res_match_template_RED_BoundingBox.tiff', r_box_image)
    # # cv2.waitKey(0)

    #### Recognizing Green Square
    # res_G = cv2.matchTemplate(image=image,templ= kernel_g,method=5)
    # cv2.imwrite('res_match_template_G.tiff', res_G)
    # min_val_G, max_val_G, min_loc_G, max_loc_G = cv2.minMaxLoc(res_G)
    # # print('min_val_G')
    # # print(min_val_G)
    # # print('max_val_G')
    # # print(max_val_G)
    # print('min_loc_G')
    # print(min_loc_G)
    # print('max_loc_G')
    # print(max_loc_G)
    #
    # # Drawing Bounding Box around detected shape
    # # determine the starting and ending (x, y)-coordinates of the bounding box
    # # From: https://www.pyimagesearch.com/2021/03/22/opencv-template-matching-cv2-matchtemplate/
    # (startX_G, startY_G) = max_loc_G
    # endX_G = startX_G + kernel_g.shape[1]
    # endY_G = startY_G + kernel_g.shape[0]
    #
    # # draw the bounding box on the image
    # g_box_image = cv2.rectangle(image, (startX_G, startY_G), (endX_G, endY_G), (0, 255, 0), 3)
    # # show the output image
    # # cv2.imshow("Output based on matchTemplate", g_box_image)
    # cv2.imwrite('res_match_template_GREEN_BoundingBox.tiff', g_box_image)
    # # cv2.waitKey(0)

'''
    cv::TemplateMatchModes 
    cv::TM_SQDIFF = 0,
    cv::TM_SQDIFF_NORMED = 1,
    cv::TM_CCORR = 2,
    cv::TM_CCORR_NORMED = 3,
    cv::TM_CCOEFF = 4,
    cv::TM_CCOEFF_NORMED = 5
'''
#### Using Bounding-Box Coordinates to get orientation of the board
# note this likely will be done in a separate function but test it here
# center_B = (np.subtract(max_loc_B[0], min_loc_B[0]),np.subtract(max_loc_B[1],min_loc_B[1]))
# center_G = np.subtract(max_loc_G, min_loc_G)
# center_R = np.subtract(max_loc_R, min_loc_R)

# print('Center_B Square:')
# print(center_B)
# shapeDetect.drawAxis()

if __name__ == '__main__':
    print("Your OpenCV version is: " + cv2.__version__)  
    image = cv2.imread("images_10x10/test_box_50x50.tif")
    # image = cv2.imread("tic_tac_toe_images/twistCorrectedColoredSquares_Color.tiff")

    # image = cv2.imread("images_20x20/test_box_20x20.tif")
    # image = image[:, :, 0]
    kernel_runner(image.copy())