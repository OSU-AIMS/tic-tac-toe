#!/usr/bin/env python

import rospy
import numpy as np 
from tictactoe_movement import TICTACTOE_MOVEMENT



def main():
    #SETUP
    rospy.init_node('tictactoe_test', anonymous=False)

    movement = TICTACTOE_MOVEMENT()

    print("scan Position")
    movement.scanPosition()


    print("X Pickup")
    movement.xPickup(0)



if __name__ == '__main__':
    main()