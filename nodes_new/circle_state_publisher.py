#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2022, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: LuisC18


#####################################################
## IMPORTS
import sys
import os

ttt_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_2_nodes = ttt_pkg + '/nodes'
path_2_scripts = ttt_pkg + '/scripts'

sys.path.insert(1, path_2_nodes)
sys.path.insert(1, path_2_scripts)

import rospy
import tf2_ros
import tf2_msgs.msg

# ROS Data Types
from std_msgs.msg import ByteMultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

# Custom Tools
# from Realsense_tools import *
from transformations import *
from toolbox_shape_detector import *
from cv_bridge import CvBridge, CvBridgeError

# System Tools
import time
from math import pi, radians, sqrt

# Ref
# http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20listener%20%28Python%29
# http://docs.ros.org/en/jade/api/geometry_msgs/html/msg/Transform.html



def findDis(pt1x, pt1y, pt2x, pt2y):
	# print('Entered Rectangle_support: findDis function')
	x1 = float(pt1x)
	x2 = float(pt2x)
	y1 = float(pt1y)
	y2 = float(pt2y)
	dis = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (0.5)
	return dis

def prepare_tiles():
	# Values for 3D printed tictactoe board
	centerxDist = 0.0635
	# centeryDist = -0.0635
	centeryDist = 0.0635 
	# removing negative sign fixed board variable, so computer correctly stores location of O blocks

	pieceHeight = 0.03

	"""
	tictactoe board order assignment:
	[0 1 2]
	[3 4 5]
	[6 7 8]
	""" 
	tf = transformations()
	# centers =[[-centerxDist ,centeryDist,pieceHeight],[0,centeryDist,pieceHeight],[centerxDist,centeryDist,pieceHeight],
	# 					[-centerxDist,0,pieceHeight],[0,0,pieceHeight],[centerxDist,0,pieceHeight],
	# 					[-centerxDist,-centeryDist,pieceHeight],[0,-centeryDist,pieceHeight],[centerxDist,-centeryDist,pieceHeight]]
	centers =[[centerxDist ,centeryDist,pieceHeight],[0,centeryDist,pieceHeight],[-centerxDist,centeryDist,pieceHeight],
						[centerxDist,0,pieceHeight],[0,0,pieceHeight],[-centerxDist,0,pieceHeight],
						[centerxDist,-centeryDist,pieceHeight],[0,-centeryDist,pieceHeight],[-centerxDist,-centeryDist,pieceHeight]]

	tictactoe_center_list = np.array(centers,dtype=np.float)
	rot_default = np.identity((3))
	new_list = []

	for vector in tictactoe_center_list:
		item = np.matrix(vector)
		new_list.append( tf.generateTransMatrix(rot_default, item) )

	return new_list

class circle_state_publisher():
	"""
	 Custom tictactoe publisher class that finds circles on image and identifies if/where the circles are on the board.
	"""

	def __init__(self, circle_state_annotation, circle_board_state,tfBuffer):

		# Inputs

		self.circle_state_annotation = circle_state_annotation
		self.circle_board_state = circle_board_state
		self.tfBuffer = tfBuffer
		self.tf = transformations()
		self.shapeDetect = TOOLBOX_SHAPE_DETECTOR()
		# camera_tile_annotation: publishes the numbers & arrows displayed on the image

		# Tools
		self.bridge = CvBridge()

	def runner(self, data):
		"""
		Callback function for image subscriber
		:param data: Camera data input from subscriber
		"""
		try:
			camera2board = self.tfBuffer.lookup_transform('ttt_board', 'camera_link', rospy.Time())
			camera2board_pose = [ 
				camera2board.transform.translation.x,
				camera2board.transform.translation.y,
				camera2board.transform.translation.z,
				0,
				0,
				0,
				0,
				# camera2board.transform.rotation.w,
				# camera2board.transform.rotation.x,
				# camera2board.transform.rotation.y,
				# camera2board.transform.rotation.z
				]
			# 2/7: Still haven't fixed rotation of numbers printed on TTT board in RQT
			
			board_tiles = prepare_tiles()

			tf_camera2board = self.tf.quant_pose_to_tf_matrix(camera2board_pose)

			tf_camera2tiles = self.tf.convertPath2FixedFrame(board_tiles,tf_camera2board)

			xyList = [[] for i in range(9)]
			scale = 1.14 / 1280

			# Convert Image to CV2 Frame
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
			img = cv_image.copy()

			for i in range(9):
				xyzCm = (tf_camera2tiles[i][0:2, 3:4])  # in cm
				x = xyzCm[0] / scale + 640
				y = xyzCm[1] / scale + 360  # in pixels

				xyList[i].append(int(x))
				xyList[i].append(int(y))
				cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
							(0, 0, 0),
							2)

			

			board = [0,0,0,0,0,0,0,0,0]
			# array for game board 0 -> empty tile, 1 -> X, -1 -> O
			
			centers, circles_img = self.shapeDetect.detectCircles(img, radius=10, tolerance=5)
			# 2/7: currently having issues detecting 2 circles.
			# TTT.py won't proceed b/c of this issue ^
			# boardCount0 = 1 but the computer only detects 2
			
			closest_square = [0, 0, 0, 0, 0, 0, 0, 0, 0]
			# ^^^ Is this used anywhere?
			
			# for each circle found
			for i in range(len(centers)):
				distanceFromCenter = findDis(centers[i][0], centers[i][1], xyList[4][0], xyList[4][1])

				if distanceFromCenter < 160:  # 100 * sqrt2
					closest_index = None
					closest = 10000
					for j in range(9):
						distance = findDis(centers[i][0], centers[i][1], xyList[j][0], xyList[j][1])

						if distance < 60 and distance < closest:
							# this creates a boundary just outside the ttt board of 40 pixels away from each tile
							# any circle within this boundary is likely to be detected as a piece in one of the 9 tiles
							closest = distance
							closest_index = j
					
					if closest_index is not None:
						board[closest_index] = -1
						cv2.circle(img, centers[i], 15, (0, 200, 40), 7)

						print("Circle {} is in tile {}.".format(i, closest_index))
					else:
						print("Circle {} is not on the board".format(i))

			print('Physical Board: ', board)
			# 2/7: max number of circles recognized is 2. 
			# Can recogniize 3 but there is flickering in detection
			# 4 circles not recognized
			for i in range(9):
				annotated_img = cv2.putText(img, str(i), (int(xyList[i][0]), int(xyList[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
							(0, 0, 0),
							2)
			
			try:
				msg_img = self.bridge.cv2_to_imgmsg(annotated_img, 'bgr8')
			except CvBridgeError as e:
				print(e)

			# Publish
			self.circle_state_annotation.publish(msg_img)

			msg_circle = ByteMultiArray()
			msg_circle.data = board
			self.circle_board_state.publish(msg_circle)
			rospy.loginfo(msg_circle)

		except rospy.ROSInterruptException:
			exit()
		except KeyboardInterrupt:
			exit()
		except CvBridgeError as e:
			print(e)


#####################################################
# MAIN
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

	pub_circle_board_state = rospy.Publisher("circle_board_state", ByteMultiArray, queue_size=20)

	# Setup Listeners
	tfBuffer = tf2_ros.Buffer()
	listener = tf2_ros.TransformListener(tfBuffer)
	cs_callback = circle_state_publisher(pub_circle_state_annotation,pub_circle_board_state,tfBuffer)
	image_sub = rospy.Subscriber("/camera/color/image_raw", Image, cs_callback.runner)

	# Auto-Run until launch file is shutdown
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")


if __name__ == '__main__':
	main()
