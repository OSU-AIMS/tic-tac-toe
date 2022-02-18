#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2021, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: LuisC18

## IMPORTS
import tf2_ros
import tf2_msgs.msg

## ROS Data Types
from std_msgs.msg import ByteMultiArray
from geometry_msgs.msg import PoseArray

import rospy
import numpy as np



def generateTransMatrix( matr_rotate, matr_translate):
        """
        Convenience Function which accepts two inputs to output a Homogeneous Transformation Matrix
        Intended to function for 3-dimensions frames ONLY
        :param matr_rotate: 3x3 Rotational Matrix
        :param matr_translate: 3x1 Translation Vector (x;y;z)
        :return Homogeneous Transformation Matrix (4x4)
        """

        ## If Translation Matrix is List, Convert
        if type(matr_translate) is list:
            matr_translate = np.matrix(matr_translate)
            # print("Changed translation vector from input 'list' to 'np.matrix'")           #TODO Commented out for debugging

        ## Evaluate Inputs. Check if acceptable size.
        if not matr_rotate.shape == (3, 3):
            raise Exception("Error Generating Transformation Matrix. Incorrectly sized inputs.")
        if not matr_translate.size == 3:
            raise Exception("Error Generating Transformation Matrix. Translation Vector wrong size.")

        ## Reformat Inputs to common shape
        if matr_translate.shape == (1, 3):
            matr_translate = np.transpose(matr_translate)
            # print("Transposed input translation vector")                                   #TODO Commented out for debugging

        ## Build Homogeneous Transformation matrix using reformatted inputs
        new_transformMatrix = np.zeros((4, 4))
        new_transformMatrix[0:0 + matr_rotate.shape[0], 0:0 + matr_rotate.shape[1]] = matr_rotate
        new_transformMatrix[0:0 + matr_translate.shape[0], 3:3 + matr_translate.shape[1]] = matr_translate
        new_transformMatrix[new_transformMatrix.shape[0] - 1, new_transformMatrix.shape[1] - 1] = 1

class TICTACTOE_LISTENER(object):
    def __init__(self):
        pass

    def retrievePose(self,pose_number):
            data = rospy.wait_for_message("tile_locations",PoseArray,timeout = None)

            return data.poses[pose_number]

    def circle_detect(self):
        '''
        Function detects circles on tictactoe image and returns updated circle board
         
        :param count0: int; number of O blocks expected
        :return boardO: 3x3 representation of which tiles have O's
        '''
        
        # print('expected number of Os: ', countO)
      
        # boardCountO=0  # does this set boardCount back to 0 each time?
       
        board = rospy.wait_for_message("circle_board_state", ByteMultiArray, timeout=None)
        # print(board.data,'boardmsg')
        board_np = np.array(board.data)
        # print('circle detect board_np',board_np)
        # print('circle detect board',board)


        boardCountO = np.count_nonzero(board_np==-1)

        boardO = [[board_np[0],board_np[1],board_np[2]],
                    [board_np[3],board_np[4],board_np[5]],
                    [board_np[6],board_np[7],board_np[8]]] 
        # boardO = [[board_np[0],board_np[3],board_np[6]],
        #             [board_np[1],board_np[4],board_np[7]],
        #             [board_np[2],board_np[5],board_np[8]]] 
        # print('circle detect boardO',boardO)
        boardO = np.array(boardO)

        return boardO, boardCountO

def main():
    """

    """
    # tf = transformations()

    # Setup Node
    rospy.init_node('ttt_listener', anonymous=False)
    rospy.loginfo(">> Tile Locator Node Successfully Created")
    tl = TICTACTOE_LISTENER()
    data = tl.tile_locations_listener()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()