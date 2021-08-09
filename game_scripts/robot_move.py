#!/usr/bin/env python

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '//home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/nodes')  

import rospy
from robot_support import *
import math
from math import pi, radians, sqrt
import numpy as np
from geometry_msgs.msg import TransformStamped
from transformations import transformations
import subprocess
from pyquaternion import Quaternion

# from visualizations import plot_path_vectors, plot_path_transforms

def prepare_path_tf_ready():
	"""
	Convenience Function to Convert Path from a List of xyz points to Transformation Matrices 
	:param path_list: Input must be list type with cell formatting of XYZ
	:return: List of Transformation Matrices
	"""
	# centerxDist = 0.05863
	# centeryDist = -0.05863
	centerxDist = 0.0635
	centeryDist = -0.0635
	pieceHeight = -0.03
	"""
	tictactoe board order assignment:
	[0 1 2]
	[3 4 5]
	[6 7 8]
	""" 
	tf = transformations()
	centers =[[-centerxDist ,centeryDist,pieceHeight],[0,centeryDist,pieceHeight],[centerxDist,centeryDist,pieceHeight],
						[-centerxDist,0,pieceHeight],[0,0,pieceHeight],[centerxDist,0,pieceHeight],
						[-centerxDist,-centeryDist,pieceHeight],[0,-centeryDist,pieceHeight],[centerxDist,-centeryDist,pieceHeight]]

	tictactoe_center_list = np.array(centers,dtype=np.float)
	print(tictactoe_center_list)
	rot_default = np.identity((3))
	new_list = []

	for vector in tictactoe_center_list:
		item = np.matrix(vector)
		new_list.append( tf.generateTransMatrix(rot_default, item) )

	return new_list

class tictactoeMotion():
	def __init__(self):
		self.board_center = geometry_msgs.msg.TransformStamped()
		self.rc = moveManipulator('bot_mh5l_pgn64')
		self.tf = transformations()

	def listener(self):

		tf_listener = '/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/nodes/board_center_subscriber.py'
		subprocess.call([tf_listener])

		tf_filename = 'tf_board2world.npy'
		data_list = np.load(str('/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe') + '/' + tf_filename)

		return data_list

	# def callback(self,data):
	# 	self.board_center = data.data

	def scanPos(self):
		self.rc.send_io(0) #open grippper
		joint_goal = self.rc.move_group.get_current_joint_values()

		joint_goal[0] = radians(90) #for right side of table
		joint_goal[1] = 0
		joint_goal[2] = 0
		joint_goal[3] = 0
		joint_goal[4] = 0
		joint_goal[5] = 0

		# Send action to move-to defined position
		self.rc.move_group.go(joint_goal, wait=True)

		# Calling ``stop()`` ensures that there is no residual movement
		self.rc.move_group.stop() 

	def xPickup(self,x,y):
		self.rc.send_io(0)#open gripper
		pose_higher = [x,y,0.08,.707,-.707,0,0]
		self.rc.goto_Quant_Orient(pose_higher)
		raw_input('Lower gripper...')

		pose_lower = [x,y,0.04,.707,-.707,0,0]
		self.rc.goto_Quant_Orient(pose_lower)

		self.rc.send_io(1) #close gripper
		pose_higher = [x,y,0.08,.707,-.707,0,0]
		self.rc.goto_Quant_Orient(pose_higher)

	def defineRobotPoses(self):
		tileCentersMatrices=prepare_path_tf_ready()

		# Body frame
		quant_board2world = self.listener()
		#print('quant_board2world:',quant_board2world)
		tf_board2world = self.tf.quant_pose_to_tf_matrix(quant_board2world)

		# Rotate board tile positions
		tileCenters2world = self.tf.convertPath2FixedFrame(tileCentersMatrices,tf_board2world)
		#print('after fixed frame',tileCenters2world)

		#Convert tfs to robot poses (quant)
		self.robot_poses =[]
		matr_rot = tileCenters2world[0][0:3,0:3]
		print('rotation matrix',matr_rot)

		b = Quaternion(matrix=matr_rot)

		for i in range(9):
			trans_rot= tileCenters2world[i][0:3,3:4]
			new_pose = [trans_rot[0][0],trans_rot[1][0],trans_rot[2][0],b[1],b[2],b[3],b[0]]
			# print(new_pose)
			self.robot_poses.append(new_pose)

	def moveToBoard(self,pose_number):
		self.rc.goto_Quant_Orient(self.robot_poses(pose_number))

		wpose = self.rc.move_group.get_current_pose().pose
		wpose.position.z += -0.01  # Move up (z)

		self.rc.goto_Quant_Orient(wpose)


		self.rc.send_io(0)#open gripper
		self.scanPos()

def main():
	try:
		ttt = tictactoeMotion()

		ttt.defineRobotPoses()
		# define robot poses based on current board center/camera position

		for i in range(9): # go to board position 0-8
			try:
				raw_input('>> Next Pose <enter>')
				ttt.moveToBoard(i)
				print('Moved to pose:')

			except KeyboardInterrupt:
				exit()
		ttt.scanPos()
	except rospy.ROSInterruptException:
		exit()
	except KeyboardInterrupt:
		exit()


if __name__ == '__main__':
	main()

