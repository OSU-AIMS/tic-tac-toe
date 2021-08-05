#!/usr/bin/env python

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
    outputFilePath = '/home/martinez737/tic-tac-toe_ws/src/tic_tac_toe/tf_board2world.npy'
    np.save(outputFilePath, data_list)

    rospy.loginfo(">> Service Provided: Exported Origin-Camera Transform to %s", outputFilePath)


if __name__ == '__main__':
    try:
        listener()
        # while True:
        #     listener()
        #     time.sleep(0.2)
    except KeyboardInterrupt:
        exit()

            
    