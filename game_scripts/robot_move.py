#!/usr/bin/env python
import os
import sys
# insert at 1, 0 is the script path (or '' in REPL)
# sys.path.insert(1, '//home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/nodes')  
sys.path.insert(1, '//home/khan764/tic-tac-toe_ws/src/tic-tac-toe/nodes')  

import rospy
from robot_support import *
import math
from math import pi, radians, sqrt
import numpy as np
from geometry_msgs.msg import TransformStamped
from transformations import transformations
import subprocess
from pyquaternion import Quaternion


def prepare_path_tf_ready():
	"""
	Adam Buynak's Convenience Function to Convert Path from a List of xyz points to Transformation Matrices
	:param path_list: Input must be list type with cell formatting of XYZ
	:return: List of Transformation Matrices
	"""
	# Values for paper tictactoe board
	# centerxDist = 0.05863
	# centeryDist = -0.05863

	# Values for 3D printed tictactoe board
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
	print('tictactoe_center_list:\n',tictactoe_center_list)
	rot_default = np.identity((3))
	new_list = []

	for vector in tictactoe_center_list:
		item = np.matrix(vector)
		new_list.append( tf.generateTransMatrix(rot_default, item) )

	return new_list


class tictactoeMotion:
	"""
	A class used to plan and execute robot poses for the tictactoe game.
	Finds the correct board positions based on current camera and board center topics.

	"""
	def __init__(self, planning_group='bot_mh5l_pgn64'):
		"""Loads moveManipulator class and transformations class
		:param planning_group: str planning group that should be loaded onto moveManipulator node
		"""
		self.rc = moveManipulator(planning_group)
		self.tf = transformations()
		self.robot_poses = []

	def listener(self):
		"""
		"listener" function that loads board center data TODO: reload camera subscriber as well.
		:return data_list: Outputs board_center transform in quaternion (x,y,z,w,x,y,z)
		"""
		# find tictactoe pkg dir
		tictactoe_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
		print('tictactoe_pkg in listener function',tictactoe_pkg)

		tf_listener = str(tictactoe_pkg) + '/nodes/board_center_subscriber.py'
		subprocess.call([tf_listener])

		tf_filename = 'tf_board2world.npy'
		data_list = np.load(str(tictactoe_pkg) + '/' + tf_filename)

		return data_list

	def scanPos(self):
		"""Returns robot to scan position for accurate board detection."""
		self.rc.send_io(0) 				# open gripper
		joint_goal = self.rc.move_group.get_current_joint_values()

		joint_goal[0] = radians(90) 	# right side of table
		joint_goal[1] = 0
		joint_goal[2] = 0
		joint_goal[3] = 0
		joint_goal[4] = 0
		joint_goal[5] = 0

		# Send action to move-to defined position
		self.rc.move_group.go(joint_goal, wait=True)

		# Calling ``stop()`` ensures that there is no residual movement
		self.rc.move_group.stop() 

	def xPickup(self,x,y,z = .1):
		"""
		Executes a pickup command to a specified x,y,z location (with respect to robot origin)
		Default values for z = .1 , hovers at 10cm above origin plane and lowers to 5 cm below input z value.
		:param x: X position in meters
		:param y: Y position in meters
		:param z: Z position in meters.
		"""
		self.rc.send_io(0)				# open gripper
		pose_higher = [x, y, z, .707, -.707, 0, 0]
		self.rc.goto_Quant_Orient(pose_higher)
		raw_input('Lower gripper in xPickup <press enter>')

		pose_lower = [x, y, z-.07, .707, -.707, 0, 0]
		self.rc.goto_Quant_Orient(pose_lower)

		self.rc.send_io(1) 				# close gripper
		pose_higher = [x, y, z, .707, -.707, 0, 0]
		self.rc.goto_Quant_Orient(pose_higher)

	def defineRobotPoses(self):
		"""
		Updates all nine robot poses for the nine grid board centers. Should be called before every robot move.
		"""
		tileCentersMatrices = prepare_path_tf_ready()
		# print('tileCentersMatrices',tileCentersMatrices)

		# Body frame
		quant_board2world = self.listener()
		print('Passed self.listener')
		tf_board2world = self.tf.quant_pose_to_tf_matrix(quant_board2world)

		# Rotate board tile positions
		tileCenters2world = self.tf.convertPath2FixedFrame(tileCentersMatrices,tf_board2world)

		# Convert tfs to robot poses (Quat)
		matr_rot = tileCenters2world[0][0:3, 0:3]
		print('rotation matrix', matr_rot)

		b = Quaternion(matrix=matr_rot)

		for i in range(9):
			trans_rot = tileCenters2world[i][0:3, 3:4]
			# print('Trans_rot in defineRobotPoses',trans_rot)
			new_pose = [trans_rot[0][0], trans_rot[1][0], trans_rot[2][0], b[1], b[2], b[3], b[0]]
			# print(new_pose)
			self.robot_poses.append(new_pose)

	def moveToBoard(self, pose_number, update=True):
		"""
		Executes robot's move by placing X on board.
		:param pose_number: Input board center location...
		[0 1 2]
		[3 4 5]
		[6 7 8]
		:param update: Whether the robot poses should be update before every move.
		True is default for tictactoe game loop, false is for demonstration purposes.
		"""
		print('inside robot_move moveToBoard')
		if update is True:
			print('update is True')
			self.defineRobotPoses()

		print('self.robot_poses',self.robot_poses)
		print('Pose number:',pose_number)
		self.rc.goto_Quant_Orient(self.robot_poses[pose_number])

		wpose = self.rc.move_group.get_current_pose().pose
		wpose.position.z += -0.01  # Move down (z)

		self.rc.goto_Quant_Orient(wpose)
		raw_input('Open gripper <press enter>')
		self.rc.send_io(0)		# open gripper
		self.scanPos()


def main():
	# DEMONSTRATION
	# Will scan the board once and run through the 9 board center positions

	try:
		ttt = tictactoeMotion()  # initiate movement class, starts manipulator node

		for i in range(9):       # go to board positions 0-8
			try:
				raw_input('>> Next Pose <enter>')
				print('i = ',i)
				ttt.moveToBoard(i,update=True)#False)
				print('Moved to pose:')

			except KeyboardInterrupt:
				exit()
		ttt.scanPos()			# return to scan position
	except rospy.ROSInterruptException:
		exit()
	except KeyboardInterrupt:
		exit()


if __name__ == '__main__':
	main()

