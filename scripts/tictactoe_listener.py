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

def generateTransMatrix(self, matr_rotate, matr_translate):
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

    def ttt_board_origin_listener(self):
            ttt_board_data= self.tfBuffer.lookup_transform('base_link', 'ttt_board', rospy.Time(0))

            ttt_board_data_list = [ 
                data.transform.translation.x,
                data.transform.translation.y,
                data.transform.translation.z,
                data.transform.rotation.w,
                data.transform.rotation.x,
                data.transform.rotation.y,
                data.transform.rotation.z
                ]

            return ttt_board_data_list

    def tile_locations(self):
            """
            tictactoe board order assignment:
            [0 1 2]
            [3 4 5]
            [6 7 8]
            """ 

            # Values for 3D printed tictactoe board
            centerxDist = 0.0635
            centeryDist = -0.0635
            pieceHeight = 0.03
          
            centers =[[-centerxDist ,centeryDist,pieceHeight],[0,centeryDist,pieceHeight],[centerxDist,centeryDist,pieceHeight],
                     [-centerxDist,0,pieceHeight],[0,0,pieceHeight],[centerxDist,0,pieceHeight],
                     [-centerxDist,-centeryDist,pieceHeight],[0,-centeryDist,pieceHeight],[centerxDist,-centeryDist,pieceHeight]]

            tictactoe_tile_center_list = np.array(centers,dtype=np.float)
            rot_default = np.identity((3))
            tfs_of_tiles = []

            for vector in tictactoe_center_list:
                item = np.matrix(vector)
                tfs_of_tiles.append( generateTransMatrix(rot_default, item) )

            return tfs_of_tiles

    def defineRobotPoses(self, tfs_of_tiles, ttt_board_origin_tf):
        """
        Updates all nine robot poses for the nine grid board centers. Should be called before every robot move.
        """
        self.robot_poses = []
        tfs_of_tiles = self.tile_locations
        
        # Body frame
        quant_board2world = self.ttt_board_origin_tf()
        tf_board2world = self.tf.quant_pose_to_tf_matrix(quant_board2world)

        # Rotate board tile positions
        tileCenters2world = self.tf.convertPath2FixedFrame(tileCentersMatrices, tf_board2world)

        # Convert tfs to robot poses (Quat)
        matr_rot = tileCenters2world[0][0:3, 0:3]
        # print('rotation matrix', matr_rot)

        b = Quaternion(matrix=matr_rot)

        for i in range(9):
            trans_rot = tileCenters2world[i][0:3, 3:4]
            # print('Trans_rot in defineRobotPoses',trans_rot)
            new_pose = [trans_rot[0][0], trans_rot[1][0], trans_rot[2][0], .707, -.707, 0, 0]
            # print('New Pose:',new_pose)
            self.robot_poses.append(new_pose)