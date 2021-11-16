#!/usr/bin/env python


## DEVELOPMENT CODE
## Used to understand how ImageConvolution works in OpenCV



## Imports
import cv2
import numpy as np



def kernel_runner(image):

    # Create kernel
    kernel_size = 4
    kernel = np.dstack((255 * np.ones((kernel_size, kernel_size, 1), dtype='uint8'),np.zeros((kernel_size, kernel_size, 3), dtype='uint8'))) # format: BGR
    kernel_b = 255 * np.ones((kernel_size, kernel_size), dtype='uint8')


    # Method: Filter2D
    # dest = image.copy()
    # result = cv2.filter2D(src=image, dst=dest, ddepth=3, kernel=kernel_b)
    # cv2.imshow("heatmap", result)
    # cv2.imshow("Dest", dest)
    # cv2.imwrite("dest.tif", result)
    # cv2.imwrite("result.tiff", result)


    # Method: Template Matching
    # image = image[:, :, 0]
    resb = cv2.matchTemplate(image=image[:, :, 0], templ=kernel_b, method=1)
    resg = cv2.matchTemplate(image=image[:, :, 1], templ=kernel_b, method=1)
    resr = cv2.matchTemplate(image=image[:, :, 2], templ=kernel_b, method=1)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)


    # Filter Results
    result = (resb*255) - (resg*255) - (resr*255)
    flag_negatives = result < 0
    result[flag_negatives] = 0
    cv2.imshow('Result', result/255)
    cv2.imwrite("res.tiff", result)

    cv2.waitKey(0)



if __name__ == '__main__':

    image = cv2.imread("image2.png")
    # image = image[:, :, 0]
    kernel_runner(image.copy())


