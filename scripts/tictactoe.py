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

import tictactoe_minimax as ai_turn
#^^Script from https://github.com/aschmid1/TicTacToeAI/blob/master/tictactoe.py



import rospy
import tf2_ros
import tf2_msgs.msg
import random


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
        countO = 1  # Number of O blocks - Humans (human goes first, so count = 1)
        countX = 0  # Number of X blocks - Robots
    	
        while True:
            boardMatrix =[
                ["","",""],
                ["","",""],
                ["","",""], ]
            # boardlist = []
    		#FUNCTION CALLS

            # wait_for_Messagefrom circle_state
            print('Before Circle Detect:')
            raw_input('Human turn: Place O <press enter>')
            boardO, boardCountO = listener.circle_detect()

            # print('Passed boardO, boardCountO = listener.circle_detect()')
            print('boardCountO: computer detected circles:',boardCountO)
            print('count0: No. of Circles in reality:',countO)
            print('Top Left of Board is Green')
            while boardCountO != countO:
                print('Inside while loop for counting Os')
                print('boardCountO',boardCountO)
                print('count0',countO)
                boardO, boardCountO = listener.circle_detect()
            
            
            board = computer.combine_board(boardO,board)
            print('After Computer.combine_board')
            print('Humans: -1 (O piece)')
            print('Computer: 1 (X piece)')

            computer.render(board)
            computer.Evaluate_Game(board) # this was commented out earlier? Why?

            for row in range(3): # range(3) = range(0,3)
                for col in range(3):
                    if board[row][col] == -1:
                        boardMatrix[row][col] = ("O")
                    elif board[row][col] == 1:
                        boardMatrix[row][col] = ("X")
                    else:
                        boardMatrix[row][col] = (" ")

            # FIND AI MOVE    
            # Note: ai_turn.move is high difficulty. Only able to tie or I suck at tic-tac-toe
            # Note: ai_turn.competent_move is low difficulty. easier to beat
            m = ai_turn.move(boardMatrix,"X") 
            print('m = ai_turn.move',m[0],m[1])

            if m == -1:
                # in ai_turn.move when no moves are available it returns None,0 originally
                # but changed it to -1
                computer.Evaluate_Game(board)
            else:
                board[m[0]][m[1]] = 1
                    # Obtain Pose number for robot motion
                if m[0] == 0:
                    pose_number = m[1]
                if m[0] == 1:
                    pose_number = 3 + m[1]
                if m[0] == 2:
                    pose_number = 6 + m[1]
        
            


            place_pose = listener.retrievePose(pose_number)

            # EXECUTE AI MOVE
            raw_input('To attempt to get X{} <press enter>'.format(countX))
            movement.xPickup(countX)
            raw_input('To attempt to get pose {} <press enter>'.format(pose_number))
            movement.placePiece(place_pose)

            raw_input('Return to scan position <press enter>')
            movement.scanPosition()

            computer.render(board)
            computer.Evaluate_Game(board)

            # Add +1 to O & X count 		
            countO += 1
            countX += 1

    except rospy.ROSInterruptException:
        exit()
    except KeyboardInterrupt:
        exit()

if __name__ == '__main__':
    main()