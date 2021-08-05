#!/usr/bin/env python

import sys
from std_msgs.msg import String
import cv2
from scan_board import *
from tictactoe_brain import *
import rospy
from PIL import Image # used for image rotation
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import subprocess
from scipy import ndimage


dXO = detectXO()
brain = BigBrain()
boardPoints=[]

board = [
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
] # array for game board 
boardCode = [
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
] # array for code to know which player went whree
  # Human: -1 (circles)
  # Computer: +1 (X's)
  # board filled with -1 & +1
countO=0

class PlayGame():

  def __init__(self):
   dXO = detectXO()
   brain = BigBrain()
   self.bridge = CvBridge()


  def listener(self):
    self.image_pub = rospy.Publisher("image_topic",Image,queue_size=20)

    self.bridge = CvBridge()
    data = rospy.wait_for_message("/camera/color/image_raw",Image,timeout=None)
    cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    cv2.imshow('test',cv_image)
    cv2.waitKey(0)
    #rospy.init_node('board_image_listener', anonymous=True)
    # print('Inside Listener')
    # self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)
    # tf_filename = 'Camera_image_data.png.npy'
    # img_data = np.load(str('/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe') + '/' + tf_filename)

    return cv_image


  def listener_Angle(self):

    tf_listener = '/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/nodes/board_center_subscriber.py'
    subprocess.call([tf_listener])

    tf_filename = 'tf_board2world.npy'
    data_list = np.load(str('/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe') + '/' + tf_filename)

    _,_,zAngle = self.euler_from_quaternion(data_list[3],data_list[4],data_list[5],data_list[6]) #(w,x,y,z)
    print('Z_Angle',zAngle)

    return zAngle 

  def callback(self,data):
    try:
      print('Callback:inside try ')
      self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      # cv_image = cv2.resize(cv_image,(640,360),interpolation = cv2.INTER_AREA)

    except CvBridgeError as e:
      print(e)
    #print(cv_image.shape)
    print('Callback: Past try & Except')
    # cv2.imshow("Image window", cv_image)
    cv2.waitKey(0)

 
  def euler_from_quaternion(self,w,x, y, z):
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
 
    return roll_x, pitch_y, yaw_z # in radians


  # Detect circles
  def circle_detect(self,countO,current_board):
    centers = []
    print('expected number of Os: ',countO)

    ## vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    ##scanning
    #robot goes into scannning position
    #robot_toPaper
    # As of 8/4/21, this^^^ is in a different script
    
    while len(centers) != countO:
      #img_precrop = imgClass.grabFrame()  -->  uncomment when using RealSense

      # cv2.imshow('Precrop',img_precrop)
      #img = imgClass.cropFrame(img_precrop) --> uncomment when using RealSense
      # cv2.imshow('Cropped Game Board',img)

      # boardImage, boardCenter, boardPoints = dXO.getContours(img)
      # cv2.imshow('boardContour',boardImage
      img = current_board # 

      centers = dXO.getCircle(img)
     
      cv2.imshow('Circles detected',img)
      #cv2.waitKey(0)
      
      #print('First array: of circles',centers[0,:])
      #print('First x of 1st array:',centers[1][0])
      #print('Length of CentersLIst',len(centers))
      #  #length = 5 for max

      ## ALL THE NUMBERS HERE WILL CHANGE B/C Board can now move & rotate
      ## Unless you move image to the blue tape corner each time & change the robot motion accordingly
      for i in range(len(centers)):
        # print('i:',i) # starts at 0
        if 7 < centers[i][0] < 54 and 13 < centers[i][1] < 60:
          print('aT top left square')
          board[0][0]='O'
          boardCode[0][0]= -1
          cv2.rectangle(img,(7,13),(54,60),(0,255,0),1)

        elif 62 < centers[i][0] < 109 and 12 <= centers[i][1] < 57:
          print('At top middle')
          board[0][1]='O'
          boardCode[0][1]= -1
          cv2.rectangle(img,(62,12),(109,57),(0,255,0),1)

        elif 115 < centers[i][0] < 164 and 12 <= centers[i][1] < 57:
          print('At top right')
          board[0][2]='O'
          boardCode[0][2]= -1
          cv2.rectangle(img,(115,12),(164,57),(0,255,0),1)

        elif 6 < centers[i][0] < 55 and 68 <= centers[i][1] < 111:
          print('At mid left')
          board[1][0]='O'
          boardCode[1][0]= -1
          cv2.rectangle(img,(6,68),(55,111),(0,255,0),1)

        elif 62 < centers[i][0] <= 108 and 66 < centers[i][1] < 109:
          print('At mid middle')
          board[1][1]='O'
          boardCode[1][1]= -1
          cv2.rectangle(img,(62,66),(108,109),(0,255,0),1)

        elif 114 < centers[i][0] <= 164 and 64 < centers[i][1] < 109:
          print('At mid right')
          board[1][2]='O'
          boardCode[1][2]= -1
          cv2.rectangle(img,(114,64),(164,109),(0,255,0),1)

        elif 5 <= centers[i][0] < 56 and 118 <= centers[i][1] < 165:
          print('At bottom left')
          board[2][0]='O'
          boardCode[2][0]= -1
          cv2.rectangle(img,(5,118),(56,165),(0,255,0),1)

        elif 63 <= centers[i][0] <= 108 and 119 < centers[i][1] < 165:
          print('At bottom middle')
          board[2][1]= 'O'
          boardCode[2][1]= -1
          cv2.rectangle(img,(63,119),(108,165),(0,255,0),1)

        elif 115 < centers[i][0] <= 167 and 117 < centers[i][1] < 163:
          print('At bottom right')
          board[2][2]='O'
          boardCode[2][2]= -1
          cv2.rectangle(img,(115,117),(167,163),(0,255,0),1)


        else:
          print('not on board')
          # 
        cv2.imshow('Tile Boundaries',img)
      

        cv2.waitKey(0)
      return boardCode, board


  # Image modification: If rotated, apply image rotation (once this works, replace it with the one in the original spot)
  def Read_Board(self,countO,current_board,board,boardCode):
    cv_image = self.listener()
    img = cv_image
    # img = self.bridge.imgmsg_to_cv2(img_data, "bgr8")
    current_board_color = current_board # frame of board taken after each move
    #cv2.imread('images/boardwO_Color.png')
    #cv2.imshow('Color current board',current_board_color)
    #current_board = cv2.cvtColor(current_board_color,cv2.COLOR_BGR2GRAY)
    # img = self.cv_image.copy()
    # cv2.imshow('GrayScale current board',img) 
    cv2.waitKey(0)


    angle = self.listener_Angle()

    rotate_img = ndimage.rotate(img,np.rad2deg(angle)) # this needs an input from GetOrientation
    cv2.imshow('Rotated image',rotate_img)
    #circle_detect(countO,rotate_img) # detects circles
    circles = self.circle_detect(countO,rotate_img) # outputs Board & BoardCode matrices
    # Need to send color image b/c scanboard.py turns image into grayscale already
    
    print('Circle Detect output',circles)
    cv2.waitKey(0)
  # # # Image modification: If rotated, apply image rotation
  # def Image_Modification(self,current_board):
  #    template = cv2.imread('images/board_sample_newRes_Color.png') # background image with no pieces
  #    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

  # # # Image subtract to see what changed
  #   return boardCode

    #output: boardCode & send to AI move so that it can choose next move

def main():
  try:
    rospy.init_node('board_image_listener', anonymous=True)
    PG = PlayGame()

    print('OpenCV version:',cv2.__version__)
    countO = 0
    current_board = cv2.imread('images/game_board_3O_Color.png') # frame of board taken after each move
    countO += 1
  #  PG.listener()
    PG.Read_Board(countO,current_board,board,boardCode)

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

