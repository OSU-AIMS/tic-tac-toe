#!/usr/bin/env python

#####################################################
#   Support Node to Output Board Position           #
#                                                   #
#   * Works Primarily in transforms                 #
#   * Relies upon camera input topic                #
#   * Publishes multiple output topics for results  #
#                                                   #
#####################################################
# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#####################################################



#####################################################
## IMPORTS
import sys
import os

ttt_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_2_game_scripts = ttt_pkg + '/game_scripts'
sys.path.insert(1, path_2_game_scripts)

import rospy

# ROS Data Types
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

# Custom Tools
  # from Realsense_tools import *
from transformations import *
from shape_detector import *
from cv_bridge import CvBridge, CvBridgeError

# System Tools
import time
from math import pi, radians, sqrt

# Ref
# http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20listener%20%28Python%29
# http://docs.ros.org/en/jade/api/geometry_msgs/html/msg/Transform.html
tf = transformations()
shapeDetect = ShapeDetector()

def findDis(pt1x,pt1y, pt2x,pt2y):
    # print('Entered Rectangle_support: findDis function')
    x1 = float(pt1x)
    x2 = float(pt2x)
    y1 = float(pt1y)
    y2 = float(pt2y)
    dis = ((x2 - x1)**2 + (y2 - y1)**2)**(0.5)
    return dis

class circle_state_publisher():
	"""
	 Custom tictactoe publisher class that finds circles on image and identifies if/where the circles are on the board.
	"""

	def __init__(self, circle_state_annotation):

		# Inputs

		self.circle_state_annotation = circle_state_annotation
		# camera_tile_annotation: publishes the numbers & arrows displayed on the image


		# Tools
		self.bridge = CvBridge()

	def runner(self, data):
		"""
		Callback function for image subscriber
		:param camera_data: Camera data input from subscriber
		"""
		try:
			board = [
			  [0, 0, 0],
			  [0, 0, 0],
			  [0, 0, 0],
			] # array for game board 
			boardCode = [
			  [0, 0, 0],
			  [0, 0, 0],
			  [0, 0, 0],
			]
			# Convert Image to CV2 Frame
			cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
			img = cv_image.copy() 

			centers = shapeDetect.detectCircle(img)

			#print('First array: of circles',centers[0,:])
			#print('First x of 1st array:',centers[1][0])
			# print('CentersList from getCircle',centers)
			#  #length = 5 for max

			## ALL THE NUMBERS HERE WILL CHANGE B/C Board can now move & rotate
			## Unless you move image to the blue tape corner each time & change the robot motion accordingly
			tictactoe_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
			tf_filename = 'tile_centers_pixel.npy'
			xyList = np.load(tictactoe_pkg + '/' + tf_filename)


			closest_index=[]
			closest_square = [0,0,0,0,0,0,0,0,0]
			for i in range(len(centers)):
				closest = 10000
			for j in range(9):
				#centers x to xList/yList centers for dist
			  
				distance = findDis(centers[i][0],centers[i][1],xyList[j][0],xyList[j][1])   
				#findDis params :(pt1x.pt1y, pt2x,pt2y)
				# print('Distance:',j,distance)
				if distance < 40 and distance < closest:
				# this creates a boundary just outside the ttt board
				# any circle within this boundary is likley to be detected as a piece in one of the 9 tiles
					closest = distance
					closest_index = j
					print('distance checker')
					# closest_square[i]= closest_index
				else:
					print('Not on board!')
			if closest_index is not None:
			  print('inside board assignent ifs')
			  # Checks which of the 9 tiles the O block is in
			  if closest_index == 0:  
				board[0][0]='O'
				boardCode[0][0]= -1
			  elif closest_index == 1:  
				board[0][1]='O'
				boardCode[0][1]= -1
			  elif closest_index == 2:  
				board[0][2]='O'
				boardCode[0][2]= -1
			  elif closest_index == 3:  
				board[1][0]='O'
				boardCode[1][0]= -1
			  elif closest_index == 4:  
				board[1][1]='O'
				boardCode[1][1]= -1
			  elif closest_index == 5:  
				board[1][2]='O'
				boardCode[1][2]= -1
			  elif closest_index == 6:  
				board[2][0]='O'
				boardCode[2][0]= -1
			  elif closest_index == 7:  
				board[2][1]='O'
				boardCode[2][1]= -1
			  elif closest_index == 8:  
				board[2][2]='O'
				boardCode[2][2]= -1

		  
				print('closest index',closest_index)
			print('Physical Board: ',board)
			print('Board Computer sees:',boardCode)
			
			try:
				msg_img = self.bridge.cv2_to_imgmsg(img, 'rgb8')
			except CvBridgeError as e:
				print(e)

			# Publish
			self.circle_state_annotation.publish(msg_img)
			#rospy.loginfo(msg)

			# cv2.imshow('CV2: Live Board', boardImage)
			# cv2.waitKey(3)


		except rospy.ROSInterruptException:
			exit()
		except KeyboardInterrupt:
			exit()
		except CvBridgeError as e:
			print(e)


#####################################################
## MAIN()
def main():
	"""
	Circle Finder.
	This script should only be launched via a launch script.
		circle_state_annotation: draws game board circles on top of camera tile annotation image
		circle_game_state: outputs game state as board code (for o's only) 
		TODO: add X detection capabilities

	"""

	# Setup Node
	rospy.init_node('circle_state', anonymous=False)
	rospy.loginfo(">> Circle Game State Node Successfully Created")


	# Setup Publishers
	pub_circle_state_annotation = rospy.Publisher("circle_state_annotation", Image, queue_size=20)

	# Setup Listeners
	cs_callback = circle_state_publisher(pub_circle_state_annotation)
	image_sub = rospy.Subscriber("/camera_tile_annotation", Image, cs_callback.runner)


	# Auto-Run until launch file is shutdown
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()

