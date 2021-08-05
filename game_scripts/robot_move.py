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
	centerxDist = 0.05863
	centeryDist = -0.05863
	pieceHeight = -0.02
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

def main():
	try:
		ttt = tictactoeMotion()
		tf = transformations()
		rc = moveManipulator('bot_mh5l_pgn64')

		# Tile centers as matrices
		tileCentersMatrices=prepare_path_tf_ready()

		# Body frame
		quant_board2world = ttt.listener()
		#print('quant_board2world:',quant_board2world)
		tf_board2world = tf.quant_pose_to_tf_matrix(quant_board2world)

		# Rotate board tile positions
		tileCenters2world = tf.convertPath2FixedFrame(tileCentersMatrices,tf_board2world)
		#print('after fixed frame',tileCenters2world)

		#Convert tfs to robot poses (quant)
		robot_poses =[]
		matr_rot = tileCenters2world[0][0:3,0:3]
		print('rotation matrix',matr_rot)

		b = Quaternion(matrix=matr_rot)

		#matrix to quat
		# t= [matr_rot[0,0],matr_rot[0,1],matr_rot[0,2],matr_rot[1,0],matr_rot[1,1],matr_rot[1,2],matr_rot[2,0], matr_rot[2,1], matr_rot[2,2]]

		# w = sqrt(t[0]+t[4]+t[8]+1)/2
		# x = sqrt(t[0]-t[4]-t[8]+1)/2
		# y = sqrt(-t[0]+t[4]-t[8]+1)/2
		# z = sqrt(-t[0]-t[4]+t[8]+1)/2
		# a = [w,x,y,z]
		# m = a.index(max(a))
		# if m == 0:
		# 		x = (t[7]-t[5])/(4*w)
		# 		y = (t[2]-t[6])/(4*w)
		# 		z = (t[3]-t[1])/(4*w)
		# if m == 1:
		# 		w = (t[7]-t[5])/(4*x)
		# 		y = (t[1]+t[3])/(4*x)
		# 		z = (t[6]+t[2])/(4*x)
		# if m == 2:
		# 		w = (t[2]-t[6])/(4*y)
		# 		x = (t[1]+t[3])/(4*y)
		# 		z = (t[5]+t[7])/(4*y)
		# if m == 3:
		# 		w = (t[3]-t[1])/(4*z)
		# 		x = (t[6]+t[2])/(4*z)
		# 		y = (t[5]+t[7])/(4*z)
		# b = [w,x,y,z]

		for i in range(9):
			trans_rot= tileCenters2world[i][0:3,3:4]
			new_pose = [trans_rot[0][0],trans_rot[1][0],trans_rot[2][0],b[1],b[2],b[3],b[0]]
			# print(new_pose)
			robot_poses.append(new_pose)
		print(robot_poses)
		#print(matr_rot)
		# robot_poses = tf.convertPath2RobotPose(tileCenters2world)


		rc.set_vel(0.1)
		rc.set_accel(0.1)

		for pose in robot_poses:
			try:
				raw_input('>> Next Pose <enter>')
				rc.goto_Quant_Orient(pose)
				print('Moved to pose:')
				print(pose)

			except KeyboardInterrupt:
				exit()
		ttt.scanPos()
	except rospy.ROSInterruptException:
		exit()
	except KeyboardInterrupt:
		exit()


if __name__ == '__main__':
	main()

