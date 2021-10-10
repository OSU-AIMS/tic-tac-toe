#!/usr/bin/env python

import sys
from std_msgs.msg import String
from tictactoe_brain import *
import rospy
import math
import subprocess
from robot_move import *

import pygame
import numpy as np

"""
A class used to plan and execute robot poses for the tictactoe game.
Finds the correct board positions based on current camera and board center topics.

"""

brain = tictactoeBrain()
motion = tictactoeMotion()

# array for code to know which player went where
# Human: -1 (circles)
# Computer: +1 (X's)`
# board filled with -1 & +1


BOARD_ROWS = 3
BOARD_COLS = 3
board = np.zeros(( BOARD_ROWS , BOARD_COLS ))

countO = 0  # Number of O blocks
countX = 0  # Number of X blocks 

player = [-1, 1]
# human: -1, computer = 1


# -------------
# FUNCTIONS
# -------------

def findDis(pt1x, pt1y, pt2x, pt2y):
    # print('Entered Rectangle_support: findDis function')
    x1 = float(pt1x)
    x2 = float(pt2x)
    y1 = float(pt1y)
    y2 = float(pt2y)
    dis = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (0.5)
    return dis



def circle_detect(countO):
    '''
    Function detects circles on tictactoe image and returns updated circle board
     
    :param count0: int; number of O blocks expected
    :return boardO: 3x3 representation of which tiles have O's
    '''
    
    # print('expected number of Os: ', countO)
    boardCountO=0

    while boardCountO != countO:
        boardCountO=0
       
        boardO = rospy.wait_for_message("circle_board_state", ByteMultiArray, timeout=None)

        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if boardO[row][col] == -1
                    boardCountO =+ 1

    return boardO
                
def x_detect(countX):
"""
Function planned to decipher X's on table and board for robotic move
params:
countX: number of X's expected on board
""" 

def combine_board(boardO,boardX)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if boardX[row][col] == 1:
                if boardO[row][col] == 0:
                    boardO[row][col] = 1
                else:
                    print("Overlapping O and X!") 
    return boardO    

def AI_move(board):
    '''
    params: 
    computer
    - refers to ai_turn from tictactoe_brain script: 
      which refers to this repo: https://github.com/Cledersonbc/tic-tac-toe-minimax
    '''

    ### Using tictactoe_brain code
    move = brain.ai_turn('X', 'O', board)  # outputs move array based on minimx
    print('MOVE: ', move)

    # modifed ai_turn to return False if no valid moves left
    if move == False:
        Evaluate_Game(move, board)

    else:
        blocksY = [.517, .5524, .5806, .609, .638, .671]
        board[move[0]][move[1]] = 1
        

        # Uncomment below after fixing orientation
        print('attempting to get X:', countX)
        Y = blocksY[countX]
        
        blocksX = -0.110
        motion.defineRobotPoses()
        raw_input('To attempt to get X <press enter>')
        motion.xPickup(blocksX, Y)
        motion.moveToBoard() #need to convert xy matrix to 1-9, how do we without 9 ifs?

    return board
     


def Evaluate_Game():
    '''
    params:
    boardCode = state of board that computer reads
    uses titactoe_brain script
    which refers to this repo: https://github.com/Cledersonbc/tic-tac-toe-minimax
    Returns game which exits the while loop in main if Game = False
    '''
    depth = len(brain.empty_cells(board))
    winner = brain.evaluate(board)

    # print('Evaluate:', winner)
    if depth == 0:

        if winner == 1:
            print('Game Over..\n Winner: A.I Computer\n\n\n')
            game = False

        elif winner == -1:
            print('Game Over..\n Winner: Human\n\n\n')
            game = False

        elif winner == 0:
            print('Tie Game!\n\n\n')
            game = False

        else:
            print('The game continues..')
            game = True
    else:
        game = True

    return game





def main():
    try:
        game = True;  # decides when game is over

        while game is True:
              
            #Define circles
            boardO = circle_detect(countO)

            #Define squares 
            # boardX = x_detect(countX)

            #Redefine board state
            # board = combine_board(boardO,boardX) ## TODO:  there is no x detection

            # Board 
            board = ai_turn(board)




        


     
    except rospy.ROSInterruptException:
        exit()
    except KeyboardInterrupt:
        exit()


if __name__ == '__main__':
    main()



## Tested on Luis VM too slow to respond (try testing at CDME)

# # initializes pygame
# pygame.init()

# # ---------
# # CONSTANTS FOR GUI
# # ---------

# WIDTH = 600
# HEIGHT = 600
# LINE_WIDTH = 15
# WIN_LINE_WIDTH = 15
# BOARD_ROWS = 3
# BOARD_COLS = 3
# SQUARE_SIZE = 200
# CIRCLE_RADIUS = 60
# CIRCLE_WIDTH = 15
# CROSS_WIDTH = 25
# SPACE = 55
# # rgb: red green blue
# RED = (255, 0, 0)
# BG_COLOR = (28, 170, 156)
# LINE_COLOR = (23, 145, 135)
# CIRCLE_COLOR = (239, 231, 200)
# CROSS_COLOR = (66, 66, 66)

# screen = pygame.display.set_mode( (WIDTH, HEIGHT) )
# pygame.display.set_caption( 'TIC TAC TOE' )
# screen.fill( BG_COLOR )

# # -------------
# # CONSOLE BOARD
# # -------------
# board = np.zeros( (BOARD_ROWS, BOARD_COLS) )
# print(board,"board")

# while game is True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             sys.exit()
        
#         if event.type == pygame.MOUSEBUTTONDOWN:
#             mouseX = event.pos[0] # x
#             mouseY = event.pos[1] # y

#             clicked_row = int(mouseY // SQUARE_SIZE)
#             clicked_col = int(mouseX // SQUARE_SIZE)

#             if available_square( clicked_row, clicked_col ):

#                 mark_square( clicked_row, clicked_col, player )
#                 # if check_win( player ):
#                 #     game_over = True
#                 player = player % 2 + 1

#                 draw_figures()

#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_r:
#                 restart()
#                 player = 1
#                 game_over = False
                
#     pygame.display.update()
# def draw_lines():
#     # 1 horizontal
#     pygame.draw.line( screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), LINE_WIDTH )
#     # 2 horizontal
#     pygame.draw.line( screen, LINE_COLOR, (0, 2 * SQUARE_SIZE), (WIDTH, 2 * SQUARE_SIZE), LINE_WIDTH )

#     # 1 vertical
#     pygame.draw.line( screen, LINE_COLOR, (SQUARE_SIZE, 0), (SQUARE_SIZE, HEIGHT), LINE_WIDTH )
#     # 2 vertical
#     pygame.draw.line( screen, LINE_COLOR, (2 * SQUARE_SIZE, 0), (2 * SQUARE_SIZE, HEIGHT), LINE_WIDTH )

# def draw_figures():
#     for row in range(BOARD_ROWS):
#         for col in range(BOARD_COLS):
#             if board[row][col] == 1:
#                 pygame.draw.circle( screen, CIRCLE_COLOR, (int( col * SQUARE_SIZE + SQUARE_SIZE//2 ), int( row * SQUARE_SIZE + SQUARE_SIZE//2 )), CIRCLE_RADIUS, CIRCLE_WIDTH )
#             elif board[row][col] == 2:
#                 pygame.draw.line( screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE), CROSS_WIDTH )    
#                 pygame.draw.line( screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE), (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), CROSS_WIDTH )

# def mark_square(row, col, player):
#     board[row][col] = player

# def available_square(row, col):
#     return board[row][col] == 0

# def restart():
#     screen.fill( BG_COLOR )
#     draw_lines()
#     for row in range(BOARD_ROWS):
#         for col in range(BOARD_COLS):
#             board[row][col] = 0
