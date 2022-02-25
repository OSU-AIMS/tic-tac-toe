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
# from tictactoe_newminimax import TicTacToe 

import test_TTT_game_script as ai_turn
#^^Script from https://github.com/aschmid1/TicTacToeAI/blob/master/tictactoe.py



import rospy
import tf2_ros
import tf2_msgs.msg
import random

def make_best_move(board, depth, player):
    """
    Controllor function to initialize minimax and keep track of optimal move choices

    board - what board to calculate best move for
    depth - how far down the tree to go
    player - who to calculate best move for (Works ONLY for "O" right now)
    """
    neutralValue = 50
    choices = []
    for move in board.availableMoves():

        board.makeMove(move, player) # returns None - which is fine
        print('TTT.py - board.board',board.board)

        moveValue = board.minimax(board, depth-1, changePlayer(player))
        # print('TTT.py - bestValue',bestValue)
        print('TTT.py - moveValue',moveValue) #always returning 100

        board.makeMove(move, " ") # Returns None - which is fine
        print('TTT.py - move after board.makeMove',move)
        # print('TTT.py - board.makeMove(move, " ")',board)

        if moveValue > neutralValue:
            choices = [move]
            print('board.board',board.board)
            print('if move>neutralValue: choices=[move]',choices)
            break
        elif moveValue == neutralValue:
            choices.append(move)
            print('elif move=neutral: choice.append(move)',choices)
    print("choices: ", choices)
    # print("len(choices): ", len(choices))

    if len(choices) > 0:
        print('random.choice(choices)',random.choice(choices))
        return random.choice(choices)
    else:
        return random.choice(board.availableMoves())
        print('random.choice(board.availableMoves())',random.choice(board.availableMoves()))



def changePlayer(player):
    """Returns the opposite player given any player"""
    if player == "X":
        return "O"
    else:
        return "X"

def main():
    #SETUP
    rospy.init_node('tictactoe', anonymous=False)
    movement = TICTACTOE_MOVEMENT()
    computer = TICTACTOE_COMPUTER()
    listener = TICTACTOE_LISTENER()
    # ai_turn = TicTacToe()


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
            # computer.Evaluate_Game(board) # this was commented out earlier? Why?

            # FIND AI MOVE
            # board, pose_number = computer.ai_turn('X', 'O', board)
            # boardlist = [board[0][0],board[0][1],board[0][2],board[1][0],board[1][1],board[1][2],board[2][0],board[2][1],board[2][2]]

            for row in range(3): # range(3) = range(0,3)
                for col in range(3):
                    if board[row][col] == -1:
                        boardMatrix[row][col] = ("O")
                    elif board[row][col] == 1:
                        boardMatrix[row][col] = ("X")
                    else:
                        boardMatrix[row][col] = (" ")
            # for row in range(3): # range(3) = range(0,3)
            #     for col in range(3):
            #         if board[row][col] == -1:
            #             boardlist.append("O")
            #         elif board[row][col] == 1:
            #             boardlist.append("X")
            #         else:
            #             boardlist.append(" ")
            # print("boardlist",boardlist)
            # ai_turn.board = boardlist # sets board variable in ai_turn to boardList
            
            # print('ai_turn.board',ai_turn.board) # being read correctly
            # pose_number = make_best_move(ai_turn,-1,"O" )
            m = ai_turn.move(boardMatrix,"X") 
            board[m[0]][m[1]] = 1

            if m[0] == 0:
                pose_number = m[1]
            if m[0] == 1:
                pose_number = 3 + m[1]
            if m[0] == 2:
                pose_number = 6 + m[1]

            # if pose_number == 0:
            #     boardlist[0]= 'X'
            #     board[0][0] = 1
            # elif pose_number == 1:
            #     boardlist[1] = 'X'
            #     board[0][1] = 1
            # elif pose_number == 2:
            #     boardlist[2] = 'X'
            #     board[0][2] = 1
            # elif pose_number == 3:
            #     boardlist[3] = 'X'
            #     board[1][0] = 1
            # elif pose_number == 4:
            #     boardlist[4] = 'X'
            #     board[1][1] = 1
            # elif pose_number == 5:
            #     boardlist[5] = 'X'
            #     board[1][2] = 1
            # elif pose_number == 6:
            #     boardlist[6] = 'X'
            #     board[2][0] = 1
            # elif pose_number == 7:
            #     boardlist[7] = 'X'
            #     board[2][1] = 1
            # elif pose_number == 8:
            #     boardlist[8] = 'X'
            #     board[2][2] = 1
            # else:
            #     print("no")


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

            # raw_input('Human turn: Place O <press enter>')
            # computer.wins(board,-1)

    		#Game over? ->  break

    except rospy.ROSInterruptException:
        exit()
    except KeyboardInterrupt:
        exit()

if __name__ == '__main__':
    main()