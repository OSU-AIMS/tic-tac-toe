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

brain = BigBrain()
PickP = tictactoeMotion()

# array for code to know which player went where
# Human: -1 (circles)
# Computer: +1 (X's)`
# board filled with -1 & +1
countO = 0

# initializes pygame
pygame.init()

# ---------
# CONSTANTS FOR GUI
# ---------

WIDTH = 600
HEIGHT = 600
LINE_WIDTH = 15
WIN_LINE_WIDTH = 15
BOARD_ROWS = 3
BOARD_COLS = 3
SQUARE_SIZE = 200
CIRCLE_RADIUS = 60
CIRCLE_WIDTH = 15
CROSS_WIDTH = 25
SPACE = 55
# rgb: red green blue
RED = (255, 0, 0)
BG_COLOR = (28, 170, 156)
LINE_COLOR = (23, 145, 135)
CIRCLE_COLOR = (239, 231, 200)
CROSS_COLOR = (66, 66, 66)

screen = pygame.display.set_mode( (WIDTH, HEIGHT) )
pygame.display.set_caption( 'TIC TAC TOE' )
screen.fill( BG_COLOR )

# -------------
# CONSOLE BOARD
# -------------
board = np.zeros( (BOARD_ROWS, BOARD_COLS) )
print(board,"board")

# board = [
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, 0, 0],
#     ]  # array for game board
boardCode = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    ]

countO = 1  # 1 block O b/c human goes first
countX = 0  # Number of X blocks used

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

def draw_lines():
    # 1 horizontal
    pygame.draw.line( screen, LINE_COLOR, (0, SQUARE_SIZE), (WIDTH, SQUARE_SIZE), LINE_WIDTH )
    # 2 horizontal
    pygame.draw.line( screen, LINE_COLOR, (0, 2 * SQUARE_SIZE), (WIDTH, 2 * SQUARE_SIZE), LINE_WIDTH )

    # 1 vertical
    pygame.draw.line( screen, LINE_COLOR, (SQUARE_SIZE, 0), (SQUARE_SIZE, HEIGHT), LINE_WIDTH )
    # 2 vertical
    pygame.draw.line( screen, LINE_COLOR, (2 * SQUARE_SIZE, 0), (2 * SQUARE_SIZE, HEIGHT), LINE_WIDTH )

def draw_figures():
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 1:
                pygame.draw.circle( screen, CIRCLE_COLOR, (int( col * SQUARE_SIZE + SQUARE_SIZE//2 ), int( row * SQUARE_SIZE + SQUARE_SIZE//2 )), CIRCLE_RADIUS, CIRCLE_WIDTH )
            elif board[row][col] == 2:
                pygame.draw.line( screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SPACE), CROSS_WIDTH )    
                pygame.draw.line( screen, CROSS_COLOR, (col * SQUARE_SIZE + SPACE, row * SQUARE_SIZE + SPACE), (col * SQUARE_SIZE + SQUARE_SIZE - SPACE, row * SQUARE_SIZE + SQUARE_SIZE - SPACE), CROSS_WIDTH )

def circle_detect():
    '''
    Function should only detect circles: CLEAN IT UP
    params: 
    count0: number of O blocks expected
    current_board= image of the current board state from camera
    countX = number of X blocks remaining
    '''
    centers = []
    
    print('expected number of Os: ', self.countO)

    boardCountO=0
    while boardCountO != self.countO:
        boardCountO=0
       
        circle_board_code = rospy.wait_for_message("circle_board_state", ByteMultiArray, timeout=None)

        for i in range(9):
            if circle_board_code[i] == -1:
                boardCountO += 1
                

    if circle_board_code[0] == -1:
        self.board[0][0] = 'O'
        self.boardCode[0][0] = -1
        mark_square(0,0)

    if circle_board_code[1] == -1:
        self.board[0][1] = 'O'
        self.boardCode[0][1] = -1

    if circle_board_code[2] == -1:
        self.board[0][2] = 'O'
        self.boardCode[0][2] = -1

    if circle_board_code[3] == -1:
        self.board[1][0] = 'O'
        self.boardCode[1][0] = -1

    if circle_board_code[4] == -1:
        self.board[1][1] = 'O'
        self.boardCode[1][1] = -1

    if circle_board_code[5] == -1:
        self.board[1][2] = 'O'
        self.boardCode[1][2] = -1

    if circle_board_code[6] == -1:
        self.board[2][0] = 'O'
        self.boardCode[2][0] = -1

    if circle_board_code[7] == -1:
        self.board[2][1] = 'O'
        self.boardCode[2][1] = -1

    if circle_board_code[8] == -1:
        self.board[2][2] = 'O'
        self.boardCode[2][2] = -1

def AI_move():
    '''
    params: 
    - boardCode: state of the board read by computer
    - refers to ai_turn from tictactoe_brain script: 
      which refers to this repo: https://github.com/Cledersonbc/tic-tac-toe-minimax
    '''

    ### Using tictactoe_brain code
    move = brain.ai_turn('X', 'O', self.boardCode)  # outputs move array based on minimx
    print('MOVE: ', move)

    # modifed ai_turn to return False if no valid moves left
    if move == False:
        # print('Inside move if-statement')
        self.Evaluate_Game(move, self.boardCode)

    else:
        blocksY = [.517, .5524, .5806, .609, .638, .671]
        self.board[move[0]][move[1]] = 'X'
        self.boardCode[move[0]][move[1]] = +1

        # Uncomment below after fixing orientation
        print('attempting to get X:', self.countX)
        Y = blocksY[self.countX]
        
        blocksX = -0.110
        PickP.defineRobotPoses()
        raw_input('To attempt to get X <press enter>')
        PickP.xPickup(blocksX, Y)

        if move[0] == 0 and move[1] == 0:
            PickP.moveToBoard(0)
        if move[0] == 0 and move[1] == 1:
            PickP.moveToBoard(1)
        if move[0] == 0 and move[1] == 2:
            PickP.moveToBoard(2)
        if move[0] == 1 and move[1] == 0:
            PickP.moveToBoard(3)
        if move[0] == 1 and move[1] == 1:
            PickP.moveToBoard(4)
        if move[0] == 1 and move[1] == 2:
            PickP.moveToBoard(5)
        if move[0] == 2 and move[1] == 0:
            PickP.moveToBoard(6)
        if move[0] == 2 and move[1] == 1:
            PickP.moveToBoard(7)
        if move[0] == 2 and move[1] == 2:
            PickP.moveToBoard(8)


def Evaluate_Game():
    '''
    params:
    boardCode = state of board that computer reads
    uses titactoe_brain script
    which refers to this repo: https://github.com/Cledersonbc/tic-tac-toe-minimax
    Returns game which exits the while loop in main if Game = False
    '''
    depth = len(brain.empty_cells(self.boardCode))
    winner = brain.evaluate(self.boardCode)
    print('Evaluate:', winner)
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

def restart():
    screen.fill( BG_COLOR )
    draw_lines()
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            board[row][col] = 0




def main():
    draw_lines()
    try:
        # rospy.init_node('board_image_listener', anonymous=True)
        
        game = True;  # decides when game is over
        pygame.display.update()

        while game is True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                

                    mouseX = event.pos[0] # x
                    mouseY = event.pos[1] # y

                    clicked_row = int(mouseY // SQUARE_SIZE)
                    clicked_col = int(mouseX // SQUARE_SIZE)

                    if available_square( clicked_row, clicked_col ):

                        mark_square( clicked_row, clicked_col, player )
                        # if check_win( player ):
                        #     game_over = True
                        # player = player % 2 + 1

                        draw_figures()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        restart()
                        player = 1
                        game_over = False


     
    except rospy.ROSInterruptException:
        exit()
    except KeyboardInterrupt:
        exit()


if __name__ == '__main__':
    main()