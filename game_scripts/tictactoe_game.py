#!/usr/bin/env python

import sys
from std_msgs.msg import String
import cv2
from shape_detector import *
from tictactoe_brain import *
import rospy
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import subprocess
from scipy import ndimage
from robot_move import *

PickP = tictactoeMotion()
shapeDetect = ShapeDetector()
brain = BigBrain()
boardPoints = []

board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]  # array for game board
boardCode = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]  # array for code to know which player went whree
# Human: -1 (circles)
# Computer: +1 (X's)`
# board filled with -1 & +1
countO = 0


def findDis(pt1x, pt1y, pt2x, pt2y):
    # print('Entered Rectangle_support: findDis function')
    x1 = float(pt1x)
    x2 = float(pt2x)
    y1 = float(pt1y)
    y2 = float(pt2y)
    dis = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (0.5)
    return dis


class PlayGame():

    def __init__(self):
        shapeDetect = ShapeDetector()
        brain = BigBrain()
        self.bridge = CvBridge()
        player = [-1, 1]
        # human: -1, computer = 1

    def listener(self):
        self.image_pub = rospy.Publisher("image_topic", Image, queue_size=20)

        self.bridge = CvBridge()
        data = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=None)
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # rospy.init_node('board_image_listener', anonymous=True)
        print('Inside Listener')
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)

        img_data = cv_image
        # tf_filename = 'Camera_image_data.png.npy' # Pulls from an image Camera_image_data.png, NOT LIVE FEED
        # img_data = np.load(str('/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe') + '/' + tf_filename)
        # img_data = np.load(str('/home/khan764/tic-tac-toe_ws/src/tic-tac-toe') + '/' + tf_filename)
        # cv2.imshow('test',img_data)
        # cv2.waitKey(0)
        return img_data

    def listener_Angle(self):

        # tf_listener = '/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/nodes/board_center_subscriber.py'
        tf_listener = '/home/khan764/tic-tac-toe_ws/src/tic-tac-toe/nodes/board_center_subscriber.py'
        subprocess.call([tf_listener])

        tf_filename = 'tf_board2world.npy'
        # data_list = np.load(str('/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe') + '/' + tf_filename)
        data_list = np.load(str('/home/khan764/tic-tac-toe_ws/src/tic-tac-toe') + '/' + tf_filename)

        _, _, zAngle = self.euler_from_quaternion(data_list[3], data_list[4], data_list[5], data_list[6])  # (w,x,y,z)
        print('Z_Angle', zAngle)

        return zAngle

    def callback(self, data):
        try:
            # print('Callback:inside try ')
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # cv_image = cv2.resize(cv_image,(640,360),interpolation = cv2.INTER_AREA)

        except CvBridgeError as e:
            print(e)
        # print(cv_image.shape)
        # print('Callback: Past try & Except')
        # cv2.imshow("Image window", cv_image)
        cv2.waitKey(0)

    def euler_from_quaternion(self, w, x, y, z):
        """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    # Detect circles
    def circle_detect(self, countO, current_board, blocks):
        '''
    Function should only detect circles: CLEAN IT UP
    params: 
    count0: number of O blocks expected
    current_board= image of the current board state from camera
    blocks = number of X blocks remaining
    '''
        centers = []
        print('expected number of Os: ', countO)

        ##scanning

        while len(centers) != countO:
            # while the number of circles detected doesn't equal the number of O blocks expected
            img = current_board  #

            centers = shapeDetect.detectCircle(img)

            cv2.imshow('Circles detected', img)
            cv2.waitKey(0)

            # print('First array: of circles',centers[0,:])
            # print('First x of 1st array:',centers[1][0])
            print('CentersList from getCircle', centers)
            #  #length = 5 for max

            ## ALL THE NUMBERS HERE WILL CHANGE B/C Board can now move & rotate
            ## Unless you move image to the blue tape corner each time & change the robot motion accordingly
            tictactoe_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            tf_filename = 'tile_centers_pixel.npy'
            xyList = np.load(tictactoe_pkg + '/' + tf_filename)

            closest_index = []
            closest_square = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(len(centers)):
                closest = 10000
                for j in range(9):
                    # centers x to xList/yList centers for dist

                    distance = findDis(centers[i][0], centers[i][1], xyList[j][0], xyList[j][1])
                    # findDis params :(pt1x.pt1y, pt2x,pt2y)
                    # print('Distance:',j,distance)
                    if distance < 40 and distance < closest:
                        # this creates a boundary just outside the ttt board
                        # any circle within this boundary is likely to be detected as a piece in one of the 9 tiles
                        closest = distance
                        closest_index = j
                        print('distance checker')
                        # closest_square[i]= closest_index
                    else:
                        print('Not on board!')
                if closest_index is not None:
                    print('inside board assignent ifs')
                    # Checks which of the 9 tiles the O block is in
                    if closest_index == 0:
                        board[0][0] = 'O'
                        boardCode[0][0] = -1
                    elif closest_index == 1:
                        board[0][1] = 'O'
                        boardCode[0][1] = -1
                    elif closest_index == 2:
                        board[0][2] = 'O'
                        boardCode[0][2] = -1
                    elif closest_index == 3:
                        board[1][0] = 'O'
                        boardCode[1][0] = -1
                    elif closest_index == 4:
                        board[1][1] = 'O'
                        boardCode[1][1] = -1
                    elif closest_index == 5:
                        board[1][2] = 'O'
                        boardCode[1][2] = -1
                    elif closest_index == 6:
                        board[2][0] = 'O'
                        boardCode[2][0] = -1
                    elif closest_index == 7:
                        board[2][1] = 'O'
                        boardCode[2][1] = -1
                    elif closest_index == 8:
                        board[2][2] = 'O'
                        boardCode[2][2] = -1

                    print('closest index', closest_index)
            print('Physical Board: ', board)
            print('Board Computer sees:', boardCode)
            self.AI_move(boardCode, blocks)
            # not really the best use of the blocks variable

    def AI_move(self, boardCode, blocks):
        '''
    params: 
    - boardCode: state of the board read by computer
    - refers to ai_turn from tictactoe_brain script: 
      which refers to this repo: https://github.com/Cledersonbc/tic-tac-toe-minimax
    '''
        ### Using tictactoe_brain code
        move = brain.ai_turn('X', 'O', boardCode)  # outputs move array based on minimx
        print('MOVE: ', move)
        # modifed ai_turn to return False if no valid moves left
        if move == False:
            print('Inside move if-statement')
            self.Evaluate_Game(move, boardCode)
        else:
            blocksY = [.517, .5524, .5806, .609, .638, .671]
            board[move[0]][move[1]] = 'X'
            boardCode[move[0]][move[1]] = +1

            # Uncomment below after fixing orientation
            print('attempting to get X:', blocks)
            Y = blocksY[blocks]
            # blocksX = -0.112
            blocksX = -0.110
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

        cv2.waitKey(0)
        return boardCode, board

    def Evaluate_Game(self, boardCode):
        '''
    params:
    boardCode = state of board that computer reads
    uses titactoe_brain script
    which refers to this repo: https://github.com/Cledersonbc/tic-tac-toe-minimax
    Returns game which exits the while loop in main if Game = False
    '''
        winner = brain.evaluate(boardCode)
        print('Evaluate:', winner)
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
        return game


def main():
    try:
        # rospy.init_node('board_image_listener', anonymous=True)
        game = True;  # decides when game is over
        PG = PlayGame()
        img = PG.listener()
        print('OpenCV version:', cv2.__version__)
        while game is True:
            countO = 1  # 1 block O b/c human goes first
            blocks = 0  # Number of X blocks used
            PG.circle_detect(countO, img, blocks)
            # current_board = cv2.imread('images/game_board_3O_Color.png') # frame of board taken after each move
            countO += 1
            blocks += 1
            print('Game Variable to decide if game continues:', game)
            print('Number of X blocks used:', blocks)
            if game == False:
                break;
            game = False

        #  PG.listener()
        # PG.Read_Board(countO,current_board,board,boardCode)

        # gameInProgress = True
        # countO = 0
        # countX = 1 # if user starts with X -> countX = 1, same with o
        # blocks = 0
        # while gameInProgress == True:
        #   # PickP.scanPos()
        #   raw_input('Press enter after Player move:')
        #   countO +=  1
        #   # RoboTurn(countO,blocks)
        #   blocks += 1

        # gameInProgress = False
        cv2.waitKey(0)

    except rospy.ROSInterruptException:
        exit()
    except KeyboardInterrupt:
        exit()


if __name__ == '__main__':
    main()

# class PlayGame():
#   '''
#   Class description: 
#   - Outputs A.I Move & End Game message
#   - Subscibes to board center & orientation to obtain center if physical board has moved or rotated
#   '''
#   def __init__(self,center_sub):
#     rospy.init_node('THE GAME_board_center_listener', anonymous=True)
#     self.center_sub = rospy.Subscriber("tictactoe_board_center",TransformStamped,self.callback)
#     self.infinity = np.inf
#     self.HUMAN = -1
#     self.COMP =  +1
#     self.board = [
#     [0, 0, 0],
#     [0, 0, 0],
#     [0, 0, 0], ]

#   def listener():
#     rospy.Subscriber("tictactoe_board_center",TransformStamped,self.callback)

#   def callback():


# # Read & Evalue Board state


#   def evaluate(self,state)
#     """
#     Function to heuristic evaluation of state.
#     :param state: the state of the current board
#     :return: +1 if the computer wins; -1 if the human wins; 0 draw
#     """
#     if self.wins(state, self.COMP):
#         score = +1
#     elif self.wins(state, self.HUMAN):
#         score = -1
#     else:
#         score = 0

#     return score


# # Recognize what changed on the board & update board variable

# # Output AI move from Minimax Algorithm
#   def AI_move():
#     pass

# # Output Game State: Win, lose, draw
#   def GameState(self,state):
#     """
#     This function test if the human or computer wins
#     :param state: the state of the current board
#     :return: True if the human or computer wins
#     """
#     return self.wins(state, self.HUMAN) or self.wins(state, self.COMP)


#   def wins(self,state, player):
#     """
#     This function tests if a specific player wins. Possibilities:
#     * Three rows    [X X X] or [O O O]
#     * Three cols    [X X X] or [O O O]
#     * Two diagonals [X X X] or [O O O]
#     :param state: the state of the current board
#     :param player: a human or a computer
#     :return: True if the player wins
#     """
#     win_state = [
#         [state[0][0], state[0][1], state[0][2]],
#         [state[1][0], state[1][1], state[1][2]],
#         [state[2][0], state[2][1], state[2][2]],
#         [state[0][0], state[1][0], state[2][0]],
#         [state[0][1], state[1][1], state[2][1]],
#         [state[0][2], state[1][2], state[2][2]],
#         [state[0][0], state[1][1], state[2][2]],
#         [state[2][0], state[1][1], state[0][2]],
#     ]
#     if [player, player, player] in win_state: 
#         return True
#     else:
#         return False
#         # if player = +1 & there are 3 in a row, Computer wins
#         # if player = -1 & there are 3 in a row, Human wins
