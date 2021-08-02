#!/usr/bin/env python

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '//home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/game_scripts')  
import rospy

# from Realsense_tools import *
from geometry_msgs.msg import TransformStamped
from transformations import *
from scan_board import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

# Ref
# http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20listener%20%28Python%29
# http://docs.ros.org/en/jade/api/geometry_msgs/html/msg/Transform.html
tf = transformations()
dXO = detectXO()


def croptoBoard(frame,center):
    #print('Entered RealsenseTools: cropFrame function\n')
    #cropped_image = frame[55:228,335:515] # frame[y,x]
    # cropped_image = frame[45:218,315:495 ] # frame[y,x]
    # cropped_image = frame[center[1]-90:center[1]+90,center[0]-90:center[0]+90] #640x480
    cropped_image = frame[center[1]-125:center[1]+125,center[0]-125:center[0]+125] #1280x720
    return cropped_image

class center_finder:

  def __init__(self):
    self.center_pub = rospy.Publisher("tictactoe_board_center",TransformStamped,queue_size=20)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)

  def detectBoard(self,frame):
    table_frame = frame.copy()
    # cv2.imshow('test',frame)
    # cv2.waitKey(0)
    # small crop to just table
    # table_frame =full_frame[0:480,0:640] # frame[y,x]
    self.boardImage, self.boardCenter, self.boardPoints= dXO.getContours(table_frame)
    # scale = .664/640 #(m/pixel)
    scale = .895/1280
    self.ScaledCenter = [0,0]
    # ScaledCenter[0] = (self.boardCenter[0]-320)*scale
    # ScaledCenter[1] = (self.boardCenter[1]-240)*scale
    self.ScaledCenter[0] = (self.boardCenter[0]-640)*scale
    self.ScaledCenter[1] = (self.boardCenter[1]-360)*scale
    print("Center of board relative to center of camera (cm):",self.ScaledCenter)


  def callback(self,data):
   
    try:
      self.boardCenter=[0,0]
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      self.detectBoard(cv_image)
      #centers = dXO.getCircle(img)
      # angle = dXO.getOrientation(self.boardPoints, self.boardImage)
      # print(np.rad2deg(angle))
      # cv2.imshow('board angle',self.boardImage)

      # boardCropped = croptoBoard(self.boardImage, self.boardCenter)
      # cv2.imshow('Cropped Board',boardCropped)
      
      boardTranslation = np.array([[self.ScaledCenter[0]],[self.ScaledCenter[1]],[0.64]])  ## depth of the table is .64 m
      boardRotation = np.identity((3))
      tf_board = tf.generateTransMatrix(boardRotation,boardTranslation) #tf_body2camera, transform from camera 
      print('before subprocess')
      import subprocess

      tf_filename = "tf_camera2world.npy"
      tf_listener = '/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/nodes/tf_origin_camera_subscriber.py'
      subprocess.call([tf_listener, '/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe', tf_filename])

      tf_list = np.load(str('/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe') + '/' + tf_filename)
      tf_camera2world = tf.quant_pose_to_tf_matrix(tf_list)
      #print('tf camera to world:',tf_camera2world)
      rot_camera_hardcode  = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

      translate            = tf_camera2world[:-1,-1].tolist()
      tf_camera2world = tf.generateTransMatrix(rot_camera_hardcode, translate)
      print('camera to robot:')
      print(np.around(tf_camera2world,2))

      tf_board2world = np.matmul(tf_camera2world,tf_board)
      print('board to robot:')
      print(np.around(tf_board2world,2))

      boardCenterPose = tf.transformToPose(tf_board2world)
      # cv2.imshow("Image window", cv_image)
      cv2.waitKey(5)
    except rospy.ROSInterruptException:
      exit()
    except KeyboardInterrupt:
      exit()
    except CvBridgeError as e:
      print(e)
    try:
      board_msg = geometry_msgs.msg.TransformStamped()
      board_msg.header.frame_id= 'OriginToBoard'
      board_msg.child_frame_id='Board'
      board_msg.transform.translation.x = tf_board2world[0][3]
      board_msg.transform.translation.y = tf_board2world[1][3]
      board_msg.transform.translation.z = tf_board2world[2][3]
      board_msg.transform.rotation.w = 0
      board_msg.transform.rotation.x = 1
      board_msg.transform.rotation.y = 0
      board_msg.transform.rotation.z = 0
      self.center_pub.publish(board_msg)
      
    except CvBridgeError as e:
      print(e)
    print('end of callback')

def main():
 

  rospy.init_node('center_finder', anonymous=True)
  cf = center_finder()
  print(">> Board Center Node Successfully Launched")
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()

