#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2022, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: LuisC18
from cv_bridge import CvBridge, CvBridgeError


class tile_locations_publisher():
    """
     Custom tictactoe publisher class that finds circles on image and identifies if/where the circles are on the board.
    """

    def __init__(self, tile_annotation, tile_locations, tfBuffer):

        # Inputs

        self.tile_annotation = tile_annotation
        self.tile_locations = tile_locations
        self.tfBuffer = tfBuffer
        # Tools
        self.bridge = CvBridge()

    def runner(self,data):

        fixed2board_tf = self.tfBuffer.lookup_transform('base_link', 'ttt_board', rospy.Time(0))
        
        #### insert tile locations relative to ttt_board

        self.tile_locations.publish(fixed2tileCenters)



        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        boardImage = cv_image.copy()
        
        xyList = [[] for i in range(9)]
        # scale = .895 / 1280  # todo: set to camera intrinsics
        scale = .664 / 640
        for i in range(9):
            xyzCm = (tileCenters2camera[i][0:2, 3:4])  # in cm
            x = xyzCm[0] / scale + 320
            y = xyzCm[1] / scale + 240  # in pixels
            xyList[i].append(int(x))
            xyList[i].append(int(y))
            cv2.putText(boardImage, str(i), (int(xyList[i][0]), int(xyList[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 0),
                        2)

        try:
            msg_img = self.bridge.cv2_to_imgmsg(boardImage, 'bgr8')
        except CvBridgeError as e:
            print(e)

        # Publish
        self.tile_annotation.publish(msg_img)



def main():
    """

    """

    # Setup Node
    rospy.init_node('tile_locations', anonymous=False)
    rospy.loginfo(">> Tile Locator Node Successfully Created")

    # Setup Publishers
    pub_tile_annotation = rospy.Publisher("tile_annotation", Image, queue_size=20)

    pub_tile_locations = rospy.Publisher("tile_locations", ByteMultiArray, queue_size=20)

    # Setup Listeners
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    tl_callback = tile_locations_publisher(pub_tile_annotation, pub_tile_locations, tfBuffer)
    image_sub = rospy.Subscriber("/camera/color/image_raw", Image, tl_callback.runner)

    # Auto-Run until launch file is shutdown
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()