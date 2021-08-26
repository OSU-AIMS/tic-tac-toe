#!/usr/bin/env python

import sys                                          # System bindings
import cv2                                          # OpenCV bindings
import numpy as np
import rospy


class ColorFinder:
  """
  Class that contains color related functions. Plan to insert dots function that finds rgb dots to find transform of desired plane.
  """

  def __init__(self):
    pass


  def deNoise(self, frame):
    # reduces noise of an image
    image_denoised = cv2.fastNlMeansDenoising(frame, None, 40, 7, 21)
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

  def colorFilter(self, image):
    """
    Filters colors a based on threshold values in function.
    @return: Grayscaled image of filtered colors.
    """
    image = image.copy()
    # split image to rgb channels
    b, g, r = cv2.split(image)

    #For black block: binary threshold any color values above "60", greater than 60 -> 255, less than 60 -> 0
    _, mask_b = cv2.threshold(b, thresh=160, maxval=255, type=cv2.THRESH_BINARY_INV)

    _, mask_g = cv2.threshold(g, thresh=160, maxval=255, type=cv2.THRESH_BINARY_INV)

    _, mask_r = cv2.threshold(r, thresh=160, maxval=255, type=cv2.THRESH_BINARY_INV)

    blank = np.zeros(image.shape[:2], dtype="uint8")
    # merge the threshold rgb channels together to create a mask of cube
    merged_mask = cv2.merge([mask_b,mask_g,mask_r])

    # grayscale merged_mask
    gray_merge = cv2.cvtColor(merged_mask, cv2.COLOR_BGR2GRAY)

    # bw merged mask
    _, mask_merge_bw = cv2.threshold(gray_merge, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

    return gray_merge


def main():
  """
  DEMONSTRATION:
  Will scan the board once and run through the 9 board center positions
  """
  try:

    for i in range(9):  # go to board positions 0-8
      try:
        raw_input('>> Next Pose <enter>')
        ttt.moveToBoard(i, update=False)
        print('Moved to pose:')

      except KeyboardInterrupt:
        exit()
    ttt.scanPos()  # return to scan position
  except rospy.ROSInterruptException:
    exit()
  except KeyboardInterrupt:
    exit()

if __name__ == '__main__':
  main()