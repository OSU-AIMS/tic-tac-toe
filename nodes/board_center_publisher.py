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
import time
from math import pi, radians, sqrt

# Ref
# http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20listener%20%28Python%29
# http://docs.ros.org/en/jade/api/geometry_msgs/html/msg/Transform.html
tf = transformations()
dXO = detectXO()

def prepare_path_tf_ready():
  """
  Convenience Function to Convert Path from a List of xyz points to Transformation Matrices 
  :param path_list: Input must be list type with cell formatting of XYZ
  :return: List of Transformation Matrices
  """
  # centerxDist = 0.05863
  # centeryDist = -0.05863
  centerxDist = 0.0635
  centeryDist = -0.0635

  pieceHeight = -0.0

  """
  tictactoe board order assignment:
  [0 1 2]
  [3 4 5]
  [6 7 8]
  """ 
  tf = transformations()
  centers =[[-centerxDist ,centeryDist,pieceHeight],[0,centeryDist,pieceHeight],[centerxDist,centeryDist,pieceHeight],
            [-centerxDist,0,pieceHeight],[0,0,pieceHeight],[centerxDist,0,pieceHeight],
            [-centerxDist,-centeryDist,pieceHeight],[0,-centeryDist,pieceHeight],[centerxDist,-centeryDist,pieceHeight]]

  tictactoe_center_list = np.array(centers,dtype=np.float)
  #print(tictactoe_center_list)
  rot_default = np.identity((3))
  new_list = []

  for vector in tictactoe_center_list:
    item = np.matrix(vector)
    new_list.append( tf.generateTransMatrix(rot_default, item) )

  return new_list

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
    #rospy.Rate(10)

  def detectBoard(self,frame):
    table_frame = frame.copy()
    # cv2.imshow('test',frame)
    # cv2.waitKey(0)
    # small crop to just table
    # table_frame =full_frame[0:480,0:640] # frame[y,x]
    self.boardImage, self.boardCenter, self.boardPoints= dXO.getContours(table_frame)
    #scale = .664/640 #(m/pixel)
    scale = .895/1280
    self.ScaledCenter = [0,0]
    # ScaledCenter[0] = (self.boardCenter[0]-320)*scale
    # ScaledCenter[1] = (self.boardCenter[1]-240)*scale
    self.ScaledCenter[0] = (self.boardCenter[0]-640)*scale
    self.ScaledCenter[1] = (self.boardCenter[1]-360)*scale
    #print("Center of board relative to center of robot (cm):",self.ScaledCenter)


  def callback(self,data):
   
    try:
      self.boardCenter=[640,360]
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      self.boardImage=cv_image.copy()
      self.detectBoard(cv_image)
      centers = dXO.getCircle(cv_image)

      #print('unordered points:',self.boardPoints)
      reorderedPoints= dXO.reorder(self.boardPoints)
      #print('reorderedPoints:',reorderedPoints)
      z_angle = dXO.newOrientation(reorderedPoints)

      angle = dXO.getOrientation(self.boardPoints, self.boardImage)
      #print('old orientation angle',np.rad2deg(angle))
      


      # boardCropped = croptoBoard(self.boardImage, self.boardCenter)
      # print(boardCropped.sh)
      # cv2.imshow('Cropped Board',boardCropped)
      
      boardTranslation = np.array([[self.ScaledCenter[0]],[self.ScaledCenter[1]],[0.655]])  ## depth of the table is .64 m

      z_orient=z_angle
      boardRotation = np.array([[math.cos(radians(z_orient)),-math.sin(radians(z_orient)),0],
                      [math.sin(radians(z_orient)),math.cos(radians(z_orient)),0],
                      [0,0,1]])
      # boardRotation = np.identity((3))

      tf_board = tf.generateTransMatrix(boardRotation,boardTranslation) #tf_body2camera, transform from camera 
      tileCentersMatrices = prepare_path_tf_ready()
      tileCenters2camera = tf.convertPath2FixedFrame(tileCentersMatrices,tf_board) # 4x4 transformation matrix
      # Columns: 0,1,2 are rotations, column: 3 is translation
      # Rows: 0,1 are x & y rotation & translation values
      xList = []
      yList = []
      scale = .895/1280
      for i in range(9): 
        xyzCm = (tileCenters2camera[i][0:2,3:4]) # in cm 
        x = xyzCm[0]/scale +640
        y = xyzCm[1]/scale +360# in pixels
        xList.append(int(x))
        yList.append(int(y))
        # print x ,y
      # print xyList

        cv2.putText(self.boardImage, str(i), (int(xList[i]), int(yList[i])), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 0, 0), 2)
      
      outputFilePathx = '/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/xList.npy'
      np.save(outputFilePathx, xList)
      outputFilePathy = '/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/yList.npy'
      np.save(outputFilePathy, yList)
      
      cv2.imshow('live board',self.boardImage)

      import subprocess
      tf_filename = "tf_camera2world.npy"
      # tf_listener = '/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/nodes/tf_origin_camera_subscriber.py'
      # p2 = time.time()
      # print("line 88:",p2-start)
      # subprocess.call([tf_listener, '/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe', tf_filename])
      # p1 = time.time()
      # print("line 90:",p1-start)
      tf_list = np.load(str('/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe') + '/' + tf_filename)

      tf_camera2world = tf.quant_pose_to_tf_matrix(tf_list)
      #print('tf camera to world:',tf_camera2world)
      rot_camera_hardcode  = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

      translate            = tf_camera2world[:-1,-1].tolist()
      tf_camera2world = tf.generateTransMatrix(rot_camera_hardcode, translate)
      # print('camera to robot:')
      # print(np.around(tf_camera2world,2))

      tf_board2world = np.matmul(tf_camera2world,tf_board)
      # print('board to robot:')
      # print(np.around(tf_board2world,2))

      boardCenterPose = tf.transformToPose(tf_board2world)
      # cv2.imshow("Image window", cv_image)
      cv2.waitKey(3)
    except rospy.ROSInterruptException:
      exit()
    except KeyboardInterrupt:
      exit()
    except CvBridgeError as e:
      print(e)
    try:
      if self.boardCenter[0]!=640 and self.boardCenter[1]!=360:
        board_msg = geometry_msgs.msg.TransformStamped()
        board_msg.header.frame_id= 'Origin'
        board_msg.child_frame_id='Board'
        board_msg.transform.translation.x = tf_board2world[0][3]
        board_msg.transform.translation.y = tf_board2world[1][3]
        board_msg.transform.translation.z = tf_board2world[2][3]

        rot_twist = tf_board2world[0:3,0:3]

        t= [rot_twist[0,0],rot_twist[0,1],rot_twist[0,2],rot_twist[1,0],rot_twist[1,1],rot_twist[1,2],rot_twist[2,0], rot_twist[2,1], rot_twist[2,2]]
        #matrix to quat
        w = sqrt(t[0]+t[4]+t[8]+1)/2
        x = sqrt(t[0]-t[4]-t[8]+1)/2
        y = sqrt(-t[0]+t[4]-t[8]+1)/2
        z = sqrt(-t[0]-t[4]+t[8]+1)/2
        a = [w,x,y,z]
        m = a.index(max(a))
        if m == 0:
            x = (t[7]-t[5])/(4*w)
            y = (t[2]-t[6])/(4*w)
            z = (t[3]-t[1])/(4*w)
        if m == 1:
            w = (t[7]-t[5])/(4*x)
            y = (t[1]+t[3])/(4*x)
            z = (t[6]+t[2])/(4*x)
        if m == 2:
            w = (t[2]-t[6])/(4*y)
            x = (t[1]+t[3])/(4*y)
            z = (t[5]+t[7])/(4*y)
        if m == 3:
            w = (t[3]-t[1])/(4*z)
            x = (t[6]+t[2])/(4*z)
            y = (t[5]+t[7])/(4*z)
        b = [w,x,y,z]

        board_msg.transform.rotation.w = b[0]
        board_msg.transform.rotation.x = b[1]
        board_msg.transform.rotation.y = b[2]
        board_msg.transform.rotation.z = b[3]
        self.center_pub.publish(board_msg)
        rospy.loginfo(board_msg)
      
    except CvBridgeError as e:
      print(e)

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

