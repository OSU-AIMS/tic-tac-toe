#!/usr/bin/env python  
import rospy

from Realsense_tools.py import *
from geometry_msgs.msg import TransformStamped



# Ref
# http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20listener%20%28Python%29
# http://docs.ros.org/en/jade/api/geometry_msgs/html/msg/Transform.html
imgClass = RealsenseTools()


if __name__ == '__main__':


    # ROS Node & Publisher to Topic
    rospy.init_node('tictactoe_board_center')
    pub = rospy.Publisher('tictactoe_board_center', TransformStamped, queue_size=10, latch=True)
    rate = rospy.Rate(1)

    print(">> Board Center Node Successfully Launched")

    while not rospy.is_shutdown():
        try:
            

        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        # Publish Collected Transform to a new topic as defined above
        rospy.loginfo(msg_new)
        pub.publish(msg_new)

        rate.sleep()


