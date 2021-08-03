#!/usr/bin/env python  
import rospy

import tf2_ros
from geometry_msgs.msg import TransformStamped



# Ref
# http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20listener%20%28Python%29
# http://docs.ros.org/en/jade/api/geometry_msgs/html/msg/Transform.html


if __name__ == '__main__':


    # ROS Node & Publisher to Topic
    rospy.init_node('tf_origin_to_camera_transform')
    pub = rospy.Publisher('tf_origin_to_camera_transform', TransformStamped, queue_size=10, latch=True)
    rate = rospy.Rate(10)

    tfBuffer = tf2_ros.Buffer(0.1)
    listener = tf2_ros.TransformListener(tfBuffer)

    print(">> TF Listener Node Successfully Launched")

    while not rospy.is_shutdown():
        try:
            msg_new = tfBuffer.lookup_transform('base_link', 'camera_color_optical_link', rospy.Time(0))

        
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        # Publish Collected Transform to a new topic as defined above
        rospy.loginfo(msg_new)
        pub.publish(msg_new)

        rate.sleep()


