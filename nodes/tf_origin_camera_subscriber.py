#!/usr/bin/env python

import rospy
import numpy as np
import sys
from geometry_msgs.msg import TransformStamped
import time


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

    #print(data_list)
    # outputFilePath = workspace + '/' + filename
    outputFilePath = '/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/tf_camera2world.npy'
    np.save(outputFilePath, data_list)
    rospy.loginfo(">> Service Provided: Exported Origin-Camera Transform to %s", outputFilePath)


if __name__ == '__main__':
    # Required Input Arguments: [absolute file path to workspace, output filename]
    # workspace = sys.argv[1]
    # filename  = sys.argv[2]

    listener()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")