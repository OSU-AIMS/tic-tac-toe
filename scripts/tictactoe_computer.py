#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2022, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: LuisC18

## IMPORTS
import numpy as np # use this for infinity
from random import choice
import platform
import time
from os import system

# infinity = np.inf
# HUMAN = -1
# COMP =  +1
# def minimax(state, depth, player):
#     """
#     AI function that choice the best move
#     :param state: current state of the board
#     :param depth: node index in the tree (0 <= depth <= 9),
#     but never nine in this case (see iaturn() function)
#     :param player: an human or a computer
#     :return: a list with [the best row, best col, best score]
#     """
#     if player == COMP:
#         best = [-1, -1, -infinity]
#     else:
#         best = [-1, -1, infinity]

#     if depth == 0 or computer.game_over(state):
#         score = computer.evaluate(state)
#         return [-1, -1, score]

#     for cell in computer.empty_cells(state):
#         x, y = cell[0], cell[1]
#         state[x][y] = player
#         score = computer.minimax(state, depth - 1, -player)
#         state[x][y] = 0
#         score[0], score[1] = x, y

#         if player == COMP:
#             if score[2] > best[2]:
#                 best = score  # max value
#         else:
#             if score[2] < best[2]:
#                 best = score  # min value

    
#     return best

class TICTACTOE_COMPUTER(object):
    """
    Class is a collection of shape detection tools based in opencv Image tools. Function image inputs require opencv image types.
    """
    def __init__(self):
        self.infinity = np.inf
        self.HUMAN = -1
        self.COMP =  +1
        self.board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0], ]

    def evaluate(self,state):
        """
        Function to heuristic evaluation of state.
        :param state: the state of the current board
        :return: +1 if the computer wins; -1 if the human wins; 0 draw
        """
        if self.wins(state, self.COMP):
            score = +1
        elif self.wins(state, self.HUMAN):
            score = -1
        elif self.wins(state, self.COMP):
            score = 0
        else:
            return 

        return score
    def Evaluate_Game(self,board):
        '''
        params:
        boardCode = state of board that computer reads
        uses titactoe_brain script
        which refers to this repo: https://github.com/Cledersonbc/tic-tac-toe-minimax
        Returns game which exits the while loop in main if Game = False
        '''
        depth = len(self.empty_cells(board)) 
        # depth is the number of empty cells which doesn't mean that the computer or human didn't win already
        # needs to change to a more specific 
        winner = self.evaluate(board)

        # print('In TTT_comp-Evaluete_Game: depth=',depth)
        # print('Evaluate:', winner)
        # if depth == 0:

        if winner == 1:
            print('Game Over..\n Winner: A.I Computer\n\n\n')
            game = False
            exit()

        elif winner == -1:
            print('Game Over..\n Winner: Human\n\n\n')
            game = False
            exit()

        elif winner == 0:
            print('Tie Game!\n\n\n')
            game = False
            exit()

        else:
            print('The game continues..')
            game = True
        # else:
        #   game = True

        return game

    def minimax(self,state, depth, player):
        """
        AI function that choice the best move
        :param state: current state of the board
        :param depth: node index in the tree (0 <= depth <= 9),
        but never nine in this case (see iaturn() function)
        :param player: an human or a computer
        :return: a list with [the best row, best col, best score]
        """
        if player == self.COMP:
            best = [-1, -1, -self.infinity]
        else:
            best = [-1, -1, +self.infinity]

        if depth == 0 or self.game_over(state):
            score = self.evaluate(state)
            return [-1, -1, score]

        for cell in self.empty_cells(state):
            x, y = cell[0], cell[1]
            state[x][y] = player
            score = self.minimax(state, depth - 1, -player)
            state[x][y] = 0
            score[0], score[1] = x, y

            if player == self.COMP:
                if score[2] > best[2]:
                    best = score  # max value
            else:
                if score[2] < best[2]:
                    best = score  # min value

        
        return best

    def empty_cells(self,state):
        """
        Each empty cell will be added into cells' list
        :param state: the state of the current board
        :return: a list of empty cells
        """
        cells = []

        for x, row in enumerate(state):
            for y, cell in enumerate(row):
                if cell == 0:
                    cells.append([x, y])

        return cells

    def ai_turn(self,c_choice, h_choice,board):
        """
        It calls the minimax function if the depth < 9,
        else it choices a random coordinate.
        :param c_choice: computer's choice X or O
        :param h_choice: human's choice X or O
        :return:
        """
        print('ai_turn: before self.board',self.board)
        self.board = board
        print('ai_turn: after self.board',self.board)
        depth = len(self.empty_cells(self.board))
        if depth == 0 or self.game_over(self.board):
            self.Evaluate_Game(self.board)
            

        # clean()
        #print(f'Computer turn [{c_choice}]')
        #render(self.board, c_choice, h_choice)

        if depth == 9: # if board is blank, randomly choose a spot
            print('Board is Blank')
            x = choice([0, 1, 2])
            y = choice([0, 1, 2])
            move = [x,y]
        else: # else apply minimax function
            move = self.minimax(self.board, depth, self.COMP)
            x, y = move[0], move[1]
            print 'IN TTT_computer: ai_turn: {}'.format((move[0],move[1]))

        self.set_move(x, y, self.COMP) # checks valid move & says computer made the move
        time.sleep(1)

        if move[0] == 0:
            pose_number = move[1]
        if move[0] == 1:
            pose_number = 3 + move[1]
        if move[0] == 2:
            pose_number = 6 + move[1]

        self.board[move[0]][move[1]] = 1

        print('IN TTT_computer ai_turn - expected board:',self.board)
        # print self.board
        return self.board, pose_number

    def game_over(self,state):
        """
        This function test if the human or computer wins
        :param state: the state of the current board
        :return: True if the human or computer wins
        """
        return self.wins(state, self.HUMAN) or self.wins(state, self.COMP)

    def valid_move(self,x, y):
        """
        A move is valid if the chosen cell is empty
        :param x: X coordinate
        :param y: Y coordinate
        :return: True if the board[x][y] is empty
        """
        if [x, y] in self.empty_cells(self.board):
            return True
        else:
            return False


    def set_move(self,x, y, player):
        """
        Set the move on board, if the coordinates are valid
        :param x: X coordinate
        :param y: Y coordinate
        :param player: the current player
        """
        if self.valid_move(x, y):
            self.board[x][y] = player
            return True
        else:
            return False

    def wins(self,state, player):
        """
        This function tests if a specific player wins. Possibilities:
        * Three rows    [X X X] or [O O O]
        * Three cols    [X X X] or [O O O]
        * Two diagonals [X X X] or [O O O]
        :param state: the state of the current board
        :param player: a human or a computer
        :return: True if the player wins
        """
        win_state = [
            [state[0][0], state[0][1], state[0][2]],
            [state[1][0], state[1][1], state[1][2]],
            [state[2][0], state[2][1], state[2][2]],
            [state[0][0], state[1][0], state[2][0]],
            [state[0][1], state[1][1], state[2][1]],
            [state[0][2], state[1][2], state[2][2]],
            [state[0][0], state[1][1], state[2][2]],
            [state[2][0], state[1][1], state[0][2]],
        ]
        if [player, player, player] in win_state:
            return True
        else:
            return False

    def render(self, state):
        """
        Print the board on console
        :param state: current state of the board
        """

        chars = {
            -1: 'O',
            +1: 'X',
            0: ' '
        }
        str_line = '-------------'

        print('\n' + str_line)

        for row in state:
            print '|',
            for cell in row:
                symbol = chars[cell]
                print("{} |".format(symbol)),
            print('\n' + str_line) 
            
    def combine_board(self,boardO,board):
        print('TTT comp: board',board)
        print('TTT comp: boardO',boardO)
        for row in range(3):
            for col in range(3):
                if boardO[row][col] == -1:
                    if board[row][col] == 0:
                        board[row][col] = -1
                    elif board[row][col] == -1:
                    	print('Previous O detected')
                    else:
                        print("Overlapping O and X!") 
        return board    

    