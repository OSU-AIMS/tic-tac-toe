#!/usr/bin/env python
#
# Software License Agreement (Apache 2.0 License)
# Copyright (c) 2021, The Ohio State University
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
#
# Author: LuisC18

import os
import sys

ttt_pkg = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_2_scripts = ttt_pkg + '/scripts'
sys.path.insert(1, path_2_scripts) 

from image_process.py import ImageProcess




def main():
    #SETUP
    rospy.init_node('tictactoe', anonymous=False)
    ImageProcess = ImageProcess()

    #MASTER LOOP
    try:
    	while true:
    		#FUNCTION CALLS





    		







    		#Game over? ->  break

    except rospy.ROSInterruptException:
        exit()
    except KeyboardInterrupt:
        exit()

if __name__ == '__main__':
    main()