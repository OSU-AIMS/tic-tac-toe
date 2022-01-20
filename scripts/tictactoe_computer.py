#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2022, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: LuisC18

## IMPORTS
class TICTACTOE_COMPUTER(object):
	"""
	Class is a collection of shape detection tools based in opencv Image tools. Function image inputs require opencv image types.
	"""
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
	    self.board = board
	    depth = len(self.empty_cells(self.board))
	    if depth == 0 or self.game_over(self.board):
	        print("GAME OVER- No valid moves left")
	        return False

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

	    self.set_move(x, y, self.COMP) # checks valid move & says computer made the move
	    time.sleep(1)

	    return move

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