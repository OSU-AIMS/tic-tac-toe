#!/usr/bin/env python

import pyrealsense2 as rs
import numpy as np
import math
import cv2
import rospy
import argparse
from std_msgs.msg import *
from os.path import join

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Starts streaming
pipeline.start(config)

# Makes list for coords and angle
index = 0
xList = []
yList = []
angleList = []
# Scaled width and height of the paper
scale = 1
# original: 2.5 with paper
hP = 216 * scale # mm
#Cardboard sheet: 218 mm
#Paper: 216 * scale  # in mm
wP = 279.4 * scale # mm
# Cardboard sheet:289 mm
# Paper: 279.4 * scale  # in mm
loop =True
try:
    while loop:
        def empty(a):
            pass
        # Determines whether coords and angle for object are consistent
        def isSame(index, xList, yList, angleList):
            if index >= 20:
                j = index - 19
            else:
                j = 0
            countX = 0
            countY = 0
            countA = 0
            while (j < (index-1)):
                if (xList[j] >= (xList[j+1] - 1) and xList[j] <= (xList[j+1] + 1)):
                    # why subtract 1 from xList[j+1]?
                    countX = countX + 1
                   # print(countX)
                if (yList[j] >= (yList[j+1] - 1) and yList[j] <= (yList[j+1] + 1)):
                    countY = countY + 1
                if (angleList[j] >= (angleList[j+1] - 3) and angleList[j] <= (angleList[j+1] + 3)):
                    countA = countA + 1
                j = j + 1
            if (countX == 18 & countY == 18 & countA == 18):
                same = True
                return same
            else:
                same = False
                return same

        # Reorders the four points of the rectangle
        def reorder(points):
            try:
                NewPoints = np.zeros_like(points)
                points = points.reshape((4, 2))
                add = points.sum(1)
                NewPoints[0] = points[np.argmin(add)]
                NewPoints[3] = points[np.argmax(add)]
                diff = np.diff(points, axis=1)
                NewPoints[1] = points[np.argmin(diff)]
                NewPoints[2] = points[np.argmax(diff)]
                return NewPoints
            except:
                return []

        # Warps the Image
        def warpImg(img, points, wP, hP):
            pad = 10
            #original: 20
            points = reorder(points)
            pts1 = np.float32(points)
            pts2 = np.float32([[0, 0], [wP, 0], [0, hP], [wP, hP]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            wint = int(wP)
            hint = int(hP)
            imgWarp = cv2.warpPerspective(img, matrix, (wint, hint))
            imgWarp = imgWarp[pad:imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]
            return imgWarp

        # Finds the distance between 2 points
        def findDis(pts1, pts2):
            x1 = float(pts1[0])
            x2 = float(pts2[0])
            y1 = float(pts1[1])
            y2 = float(pts2[1])
            dis = ((x2 - x1)**2 + (y2 - y1)**2)**(0.5)
            return dis

        # Draws x-y axis relative to object center and orientation
        def drawAxis(img, p_, q_, color, scale):
            p = list(p_)
            q = list(q_)

            angle = math.atan2(p[1] - q[1], p[0] - q[0])  #in radians
            hypotenuse = math.sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

            # Lengthens the arrows by a factor of scale
            q[0] = p[0] - scale * hypotenuse * math.cos(angle)
            q[1] = p[1] - scale * hypotenuse * math.sin(angle)
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

            # Creates the arrow hooks
            p[0] = q[0] + 9 * math.cos(angle + math.pi / 4)
            p[1] = q[1] + 9 * math.sin(angle + math.pi / 4)
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

            p[0] = q[0] + 9 * math.cos(angle - math.pi / 4)
            p[1] = q[1] + 9 * math.sin(angle - math.pi / 4)
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

        # Gets angle of object
        def getOrientation(pts, img): 
            sz = len(pts)
            data_pts = np.empty((sz, 2), dtype=np.float64)
            for i in range(data_pts.shape[0]):
                data_pts[i, 0] = pts[i, 0, 0]
                data_pts[i, 1] = pts[i, 0, 1]

            # Performs PCA analysis
            mean = np.empty((0))
            mean, eigenvectors = cv2.PCACompute2(data_pts, mean)

            # Stores the center of the object
            cntr = (int(mean[0, 0]), int(mean[0, 1]))
            # Draws the principal components
            cv2.circle(img, cntr, 3, (255, 0, 255), 2)
            p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0],
                  cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
            p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0],
                  cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
            drawAxis(img, cntr, p1, (255, 255, 0), 1)
            drawAxis(img, cntr, p2, (0, 0, 255), 5)
            angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
            return angle, cntr, mean

        def write():
            print('Creating a new file')
            path = "/home/khan764/ws_Robot/src/pick-and-place/scripts"
            name = 'disPixels.txt'  # Name of text file coerced with +.txt
            return name, path

        # Finds largest rectangular object
        def getContours(img, imgContour, minArea, filter):
            cThr = [60, 100]
            # original: [100,100]

            # try:
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
            # except:
            #     imgBlur = cv2.GaussianBlur(img,(5,5),1)
            imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
            kernel = np.ones((3, 3))
            imgDilate = cv2.dilate(imgCanny, kernel, iterations=3)
            imgThre = cv2.erode(imgDilate, kernel, iterations=2)
            #thresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 205, 1)
            cv2.imshow("img threshold",imgThre)
            #rospy.sleep(1)
            contours, _ = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            areaList = []
            approxList = []
            bboxList = []
            BiggestContour = []
            BiggestBounding =[]
            for i in contours:
                #print('Contours: ',contours)
                area = cv2.contourArea(i)
                if area > minArea:
                    cv2.drawContours(imgContour, [i], -1, (255, 0, 0), 3)
                    peri = cv2.arcLength(i, True)
                    approx = cv2.approxPolyDP(i, 0.04 * peri, True)
                    if len(approx) == filter:
                        bbox = cv2.boundingRect(approx)
                        areaList.append(area)
                        approxList.append(approx)
                        bboxList.append(bbox)
                        
            if len(areaList) != 0:
                SortedAreaList= sorted(areaList, reverse=True)
                #print("sorted area list: ",SortedAreaList)
                BiggestIndex = areaList.index(SortedAreaList[0])
                #print('BiggestIndex: ',BiggestIndex)
                BiggestContour = approxList[BiggestIndex]
                BiggestBounding = bboxList[BiggestIndex]


            return BiggestContour, BiggestBounding

            # approxlist=perimeter
        # def camera_node():
        #     pub = rospy.Publisher('positionData',String,queue_size=10)
        #     rospy.init_node('positionData', anonymous=True)
        #     rate = rospy.Rate(10) # 10hz
        #     while not rospy.is_shutdown():
        #         hello_str = "hello world %s" % rospy.get_time()
        #         rospy.loginfo(hello_str)
        #         pub.publish(hello_str)
        #         rate.sleep()

        def talker():
            pub = rospy.Publisher('Coordinates/Angle', Float64, queue_size=10)
            # is float64 right for Python 2.7? 
            rospy.init_node('Camera')
            print('Initialized Node')

            # publist = Float64()
            # a = [pub_angle,xC,yC] # storing variables in array
            # publist.data=list(a)
            # print('Made list into publist data')
            # rospy.loginfo(publist.data)
            # print(publist.data)
            # pub.publish(publist.data)

            # print('Published publlist')
            rospy.sleep(5)
            pub.publish(pub_angle)
            rospy.sleep(5)
            pub.publish(xC)
            rospy.sleep(5)
            pub.publish(yC)
            rospy.spin()
            rospy.sleep(10)
          


        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Converts images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image_preCrop = np.asanyarray(color_frame.get_data())

        #cropping the color_image to ignore table
        color_image = color_image_preCrop[60:250, 200:500]
        # 85:250, 85:220 => top left, small square of blue tarp
        # 250:500, 250:500 => bottom right
        # 100:500, 100:500 => too far to the left but good
        # 100:500, 200:500 => move further up
        # 60:400, 200:500 => move further up
        # 40:90, 200:500
        # 60:250, 200:500 => works!
        cv2.imshow("Cropped color image",color_image)

        # Applys colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))



        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('RealSense',images)



        imgCont = color_image.copy()

        #imgDepth=depth_frame.copy()
        #cv2.imshow('Depth',depth_frame)
        #cv2.imshow('Depth color',depth_colormap)
        #print('Depth Array',depth_frame)
        # Gets Contour of Paper
        # a, b, c = getContours(color_image, imgCont, minArea = 3000, filter=4)
        # ^^^^^ Looks for the paper

        cv2.imshow('RealSense',imgCont)

        # if len(a) != 0:
            # biggest = b[0]
            # # Warps Image
            # imgWarp = warpImg(imgCont, biggest, wP, hP)
            # imgCont2 = imgWarp.copy()
            # Gets Contour of Block
        bigCont, boundingBox = getContours(color_image, imgCont, minArea = 150, filter = 4)
        if len(bigCont) != 0:
            # Finds biggest contour
            cv2.polylines(imgCont,bigCont,True,(0,255,0),2)
            nPoints = (bigCont)
            if len(nPoints) != 0:
                # Locates center point, distance to edge of paper and finds angle
                NewWidth = round(findDis(nPoints[0][0]//scale, nPoints[1][0]//scale)/10,3)   #in cm
                NewHeight = round(findDis(nPoints[0][0]//scale, nPoints[2][0]//scale)/10,3)  #in cm
              #  print('New Width: ',cntr[0])
              #  print('New Height: ',cntr[2])
                cv2.arrowedLine(imgCont, (nPoints[0][0][0], nPoints[0][0][1]),(nPoints[1][0][0], nPoints[1][0][1]),
                            (255,0,255),3,8,0, 0.05)
                cv2.arrowedLine(imgCont, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                            (255,0,255),3,8,0,0.05)
                x,y,w,h = boundingBox

                angle, cntr, mean = getOrientation(nPoints, imgCont)
                y_range=range((cntr[1]-50),(cntr[1]+50),1)
                x_range=range((cntr[0]-50),(cntr[0]+50),1)
                # print('X' + str(len(x_range)) + 'Y' + str(len(y_range)))
                # y_pixels=[np.ones(len(y_range))]
                # x_pixels=[np.ones(len(x_range))]
                # dis_pixels= np.zeros((len(x_range),len(y_range)))
                # stray = []
                # i = 0
                # while(i < len(y_range)-1):
                #     j = 0
                #     while(j < len(x_range)-1):
                #         dis_pixels[j,i] = depth_frame.get_distance(x_range[j], y_range[i])
                #         var = str(dis_pixels[j,i])
                #         stray.append(var)
                #         j = j+1
                #     i = i+1
                # name, path = write()
                # file = open(join(path, name),'w')   # Trying to create a new file or open one
                # i2 = 0
                # # DO NOT RUN BELOW --> Run risk to craashing computer & creating 7.0 gB txt file
                # # while (i2 < len(stray)-1):
                # #     file.write(stray[i2] +'/n')

                # #file.write()
                # file.close()
                # print('wrote to file')

                        
                result_angle = int(np.rad2deg(angle)) # in deg
                # multiplying by negative 1 to match gripper CW(-) & CCW(+)
                xC = round(findDis((cntr[0], cntr[1]), (0, cntr[1])) / (10*scale), 3) # in cm
                yC = round(findDis((cntr[0], cntr[1]), (cntr[0], 0)) / (10*scale), 3) # in cm
                #round((hP/(10*scale)) - 
                # Makes List for coordinates and angle
                xList.append(xC)
                yList.append(yC)
                angleList.append(result_angle)
                index = index + 1

                # Draws arrows and puts text on image
                cv2.arrowedLine(imgCont, (cntr[0], cntr[1]), (cntr[0],1000), (0, 255, 0), 2, 8, 0, 0.05)
                cv2.arrowedLine(imgCont, (cntr[0],cntr[1]), (0, cntr[1]), (0, 255,0),2,8,0,0.05)
                center = [NewWidth / 2, NewHeight / 2]
                cv2.putText(imgCont, '{}cm'.format(NewWidth),(x+30, y-10), cv2.FONT_HERSHEY_PLAIN, 0.75, (0,0,0),1)
                cv2.putText(imgCont, '{}cm'.format(NewHeight),(x-70, y+h//2), cv2.FONT_HERSHEY_PLAIN, 0.75, (0,0,0),1)
                label = "  Rotation Angle: " + str(int(np.rad2deg(angle))) + " degrees" 
                #
                textbox = cv2.rectangle(imgCont, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10),
                                        (255, 255, 255), 1)
                cv2.putText(imgCont, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)
                cv2.imshow('DrawBS',imgCont)

                # Checks whether the same coords and angle are being detected consistently
                if (index >= 19):
                    print('counting')
                    same = isSame(index, xList, yList, angleList)
                    if (same == True):
                        print('********* RESULTS ***************')
                        print('Angle is ' + str(result_angle) + ' degrees [CW Positive]')
                        print('Coordinate of center is (' + str(xC) + ' , ' + str(yC) + ') cm')
                        pub_angle = int(np.rad2deg(result_angle))
                        #talker()
                        a = [str(xC),str(yC),str(result_angle)]
                        #np.savetxt('Coordinate-angle.txt', zip(a), fmt="%5.2f")
                        #file.write(str(xC)+ '/n'+str(yC)+ '/n'+str(pub_angle))
                        # name, path = write()
                        # file = open(join(path, name),'w')   # Trying to create a new file or open one
                        # file.write(a[0] +'\n' +a[1] + '\n' +a[2])
                        # file.close()
                        # print('wrote to file')
                        #with open("/home/martinez737/ws_pick_camera/Coordinate-angle.txt", "r") as f:
                            #file_content = f.read()
                            #fileList = file_content.splitlines()
                            #xCNew = float(fileList[0])
                            #yCNew = float(fileList[1])
                            #angleNew = float(fileList[2])


                        loop=False
                            
        #camera_node()
        cv2.waitKey(1)
except rospy.ROSInterruptException:
    print("ROSInterruptException")
    exit()
except KeyboardInterrupt:
    exit()
#finally:
    # print('out of loop')
    # # Stop streaming
    # pipeline.stop()
    # exit()