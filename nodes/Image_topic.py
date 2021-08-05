#!/usr/bin/env python

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from PIL import Image

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic",Image,queue_size=20)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      # cv_image = cv2.resize(cv_image,(640,360),interpolation = cv2.INTER_AREA)

    except CvBridgeError as e:
      print(e)
    #print(cv_image.shape)
    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    try:
      # imgData = (self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
      imgData = cv_image
      outputFilePath = '/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/Camera_image_data.png'

      imgData.save('/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/Camera_image_data.png')

      #np.save(outputFilePath, imgData)
    except CvBridgeError as e:
      print(e)

def main():
  
  rospy.init_node('image_converter', anonymous=True)
  ic = image_converter()
  print(">> Image Node Successfully Launched")
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()