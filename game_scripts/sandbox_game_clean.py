#!/usr/bin/env python

import sys
from std_msgs.msg import String
import cv2
from shape_detector import *
from tictactoe_brain import *
import rospy
from PIL import Image # used for image rotation
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import subprocess




class PlayGame():

	def __init__(self):
		dXO = detectXO()
		brain = BigBrain()
		self.bridge = CvBridge()


	def listener(self):
		#rospy.init_node('board_image_listener', anonymous=True)

		self.image_pub = rospy.Publisher("image_topic",Image,queue_size=20)

		self.bridge = CvBridge()
		data = rospy.wait_for_message("/camera/color/image_raw",Image,timeout=None)
		self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		# print('Listener: after subscriber')
		# cv2.imshow("Image window", self.cv_image)
		# cv2.waitKey(0)

		# tf_filename = 'Camera_image_data.png.npy'
		# img_data = np.load(str('/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe') + '/' + tf_filename)

		# return img_data

	def callback(self,data):
		print('Callback:before try ')
		try:
			print('Callback:inside try ')
			self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
			# cv_image = cv2.resize(cv_image,(640,360),interpolation = cv2.INTER_AREA)

		except CvBridgeError as e:
			print(e)
		#print(cv_image.shape)
		print('Callback: Past try & Except')
		cv2.imshow("Image window", cv_image)
		cv2.waitKey(0)



def main():
	try:
		rospy.init_node('board_image_listener', anonymous=True)
		PG = PlayGame()
		PG.listener()
		# print('OpenCV version:',cv2.__version__)
		# countO = 0
		# current_board = cv2.imread('images/game_board_3O_Color.png') # frame of board taken after each move
		# countO += 1
		# PG.listener()
	#  PG.Read_Board(countO,current_board,board,boardCode)

		cv2.waitKey(0)

	except rospy.ROSInterruptException:
		exit()
	except KeyboardInterrupt:
		exit()





if __name__ == '__main__':

	main()