#!/usr/bin/env python
import os
import rospy
import numpy as np
import sys
from geometry_msgs.msg import TransformStamped
import time


def listener():
    # Intialize New Node for Subscriber, Wait for Topic to Publish, Subscribe to Topic
    rospy.init_node('tictactoe_board_center_listener', anonymous=True)
    data = rospy.wait_for_message('tictactoe_board_center', TransformStamped, timeout=None)

    data_list = [ 
        data.transform.translation.x,
        data.transform.translation.y,
        data.transform.translation.z,
        data.transform.rotation.w,
        data.transform.rotation.x,
        data.transform.rotation.y,
        data.transform.rotation.z
        ]

    #print(data_list)
    tictactoe_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tf_filename = 'tf_board2world.npy'

    outputFilePath = tictactoe_pkg + '/' + tf_filename
    np.save(outputFilePath, data_list)

    rospy.loginfo(">> Service Provided: Exported Origin-Board Transform to %s", outputFilePath)


if __name__ == '__main__':
    try:
        listener()
        rospy.spin()
    except KeyboardInterrupt:
        exit()

            
    