#!/usr/bin/env python

import rospy
import numpy as np
import sys
from geometry_msgs.msg import TransformStamped
import time
import os


def listener():
    # Intialize New Node for Subscriber, Wait for Topic to Publish, Subscribe to Topic
    rospy.init_node('tf_origin_to_camera_listener', anonymous=True)
    data = rospy.wait_for_message('tf_origin_to_camera_transform', TransformStamped, timeout=None)
  
    data_list = [ 
        data.transform.translation.x,
        data.transform.translation.y,
        data.transform.translation.z,
        data.transform.rotation.x,
        data.transform.rotation.y,
        data.transform.rotation.z,
        data.transform.rotation.w
        ]

    tictactoe_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tf_filename = 'tf_camera2world.npy'

    outputFilePath = tictactoe_pkg + '/' + tf_filename
    np.save(outputFilePath, data_list)
    rospy.loginfo(">> Service Provided: Exported Origin-Camera Transform to %s", outputFilePath)


if __name__ == '__main__':
    # Required Input Arguments: [absolute file path to workspace, output filename]
    # workspace = sys.argv[1]
    # filename  = sys.argv[2]
    try:
        listener()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        exit()