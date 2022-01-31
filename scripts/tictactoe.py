#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2021, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: LuisC18

import os
import sys
import numpy as np 

from tictactoe_computer import TICTACTOE_COMPUTER
from tictactoe_movement import TICTACTOE_MOVEMENT
from tictactoe_listener import TICTACTOE_LISTENER

import rospy
import tf2_ros
import tf2_msgs.msg


def main():
    #SETUP
    rospy.init_node('tictactoe', anonymous=False)
    movement = TICTACTOE_MOVEMENT()
    computer = TICTACTOE_COMPUTER()
    listener = TICTACTOE_LISTENER()


    #MASTER LOOP
    try:
        movement.scanPosition()
        board = np.zeros((3,3))
        countO = 0  # Number of O blocks
        countX = 0  # Number of X blocks 
    	while True:
    		#FUNCTION CALLS

            # wait_for_Messagefrom circle_state
            boardO, boardCountO = listener.circle_detect()
            while boardCountO != countO:
                boardO, boardCountO = listener.circle_detect()

            board = computer.combine_board(boardO,board)

            # FIND AI MOVE
            board, pose_number = computer.ai_turn('X', 'O', board)
            place_pose = listener.retrievePose(pose_number)

            # EXECUTE AI MOVE
            raw_input('To attempt to get X{} <press enter>'.format(countX))
            movement.xPickup(countX)
            raw_input('To attempt to get pose {} <press enter>'.format(pose_number))
            movement.placePiece(place_pose)

            # Add +1 to O & X count 		
            countO += 1
            countX += 1

    		#Game over? ->  break

    except rospy.ROSInterruptException:
        exit()
    except KeyboardInterrupt:
        exit()

if __name__ == '__main__':
    main()