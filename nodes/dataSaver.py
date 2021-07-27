#!/usr/bin/env python

import rospy
import numpy as np
import geometry_msgs.msg

class DataSaver(object):
    """
    All data saving functions wrapped in one tool.
    """
    #Adapted from Adam Buynak's Maze Runner code


    #todo: this assumes only ONE robot planning scene. Actually using TWO scenes. DataSaver does NOT make this distinction when saving poses!!!
    #total board size 183mm x183mm
    def __init__(self, tictactoe_ws, robot, camera, board_size = [0.183, 0.183]):
        #Setup
        self.workspace = mzrun_ws
        self.robot = robot
        self.camera = camera
        self.board_size = board_size

        setup_dict = {}
        setup_dict['counter'] = []
        setup_dict['poses'] = []
        setup_dict['images'] = []
        setup_dict['maze_rotationMatrix'] = []
        setup_dict['maze_centroid'] = []
        setup_dict['scale'] = []
        setup_dict['maze_origin'] = []
        setup_dict['maze_soln_filepath'] = []
        setup_dict['tf_camera2world'] = []
        setup_dict['tf_body2camera'] = []
        setup_dict['dots'] = []

        self.all_data = setup_dict


    def capture(self, img_counter, find_maze=False):
        #path_img = self.camera.capture_singleFrame_color(self.workspace + '/' + str(img_counter))
        path_img, path_depth_npy = self.camera.capture_singleFrame_alignedRGBD(self.workspace + '/' + str(img_counter))
        pose = self.robot.lookup_pose()

        # Calculate Camera Transformation (relative to world frame)
        import subprocess
        tf_filename = "tf_camera2world.npy"
        tf_listener = os.path.dirname(self.workspace) +'/nodes/tf_origin_camera_subscriber.py'
        subprocess.call([tf_listener, self.workspace, tf_filename])

        tf_list = np.load(str(self.workspace) + '/' + tf_filename)
        tf_camera2world = quant_pose_to_tf_matrix(tf_list)


        if find_maze: 
            # Run Vision Pipeline, find Location & Rotation of Maze
            featureData_dots_filepaths = retrieve_pose_from_dream3d(self.workspace, path_img, 1000)
            centroid, rotationMatrix, scale, mazeOrigin, mazeSolutionList, tf_body2camera, dots  = characterize_maze(self.camera, self.workspace, img_id=img_counter, maze_size=self.maze_size, featureData_dots_filepaths=featureData_dots_filepaths, path_depth_npy=path_depth_npy)

            self.save_data(self.workspace, img_counter, path_img, pose, centroid, rotationMatrix, scale, mazeOrigin, mazeSolutionList, tf_camera2world, tf_body2camera, dots)
        else:
            self.save_data(self.workspace, img_counter, path_img, pose, np.array([0]), np.array([0]), 0, np.array([0]), 'N/A', tf_camera2world, np.array([0]), np.array([0]))


    def save_data(self, mzrun_ws, img_counter, path_img, pose, maze_centroid, maze_rotationMatrix, scale, mazeOrigin, mazeSolutionList, tf_camera2world, tf_body2camera, dots) :
        """
        Take Photo AND Record Current Robot Position
        :param robot:     Robot Instance
        :param camera:    Realsense Camera Instance
        :param mzrun_ws:  Local Workspace Path for Variables
        :param image_counter: Counter Input for consequitive location is saved at. 
        :return           Updated Storage Dicitonary
        """

        add_data = self.all_data

        add_data['counter'].append(img_counter)
        add_data['images'].append(path_img)
        add_data['poses'].append(json_message_converter.convert_ros_message_to_json(pose))
        add_data['maze_centroid'].append(np.ndarray.tolist(maze_centroid))
        add_data['maze_rotationMatrix'].append(np.ndarray.tolist(maze_rotationMatrix))
        add_data['scale'].append(scale)
        add_data['maze_origin'].append(mazeOrigin)
        add_data['maze_soln_filepath'].append(str(mazeSolutionList))
        add_data['tf_camera2world'].append(np.ndarray.tolist(tf_camera2world))
        add_data['tf_body2camera'].append(np.ndarray.tolist(tf_body2camera))
        add_data['dots'].append(dots)

        with open(mzrun_ws + '/camera_poses.json', 'w') as outfile:
            json.dump(add_data, outfile, indent=4)

        self.all_data = add_data

    def last_recorded(self):
        # Build Temporary Dictionary to Return latest Data

        latest_data = {
            'counter': self.all_data['counter'][-1],
            'images': self.all_data['images'][-1],
            'poses' : self.all_data['poses'][-1],
            'maze_centroid' : self.all_data['maze_centroid'][-1],
            'maze_rotationMatrix' : self.all_data['maze_rotationMatrix'][-1],
            'scale' : self.all_data['scale'][-1],
            'maze_origin' : self.all_data['maze_origin'][-1],
            'maze_soln_filepath' : self.all_data['maze_soln_filepath'][-1],
            'tf_camera2world' : self.all_data['tf_camera2world'][-1], 
            'tf_body2camera' : self.all_data['tf_body2camera'][-1],
            'dots' : self.all_data['dots'][-1],
        }

        return latest_data