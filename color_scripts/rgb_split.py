#!/usr/bin/env python


#programming_fever
import cv2
import numpy as np
import pandas as pd

img_path = 'images/red.jpeg'
img = cv2.imread(img_path)
img=cv2.resize(img,(700,500))

clicked = False
r = g = b = xpos = ypos = 0

#Reading csv file with pandas and giving names to each column
index=["color","color_name","hex","R","G","B"]
csv = pd.read_csv('colors.csv', names=index, header=None)

#function to calculate minimum distance from all colors and get the most matching color
def getColorName(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R- int(csv.loc[i,"R"])) + abs(G- int(csv.loc[i,"G"]))+ abs(B- int(csv.loc[i,"B"]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"color_name"]
    return cname

#function to get x,y coordinates of mouse double click
def draw_function(event, x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b,g,r,xpos,ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)
cv2.namedWindow('color detection by programming_fever')
cv2.setMouseCallback('color detection by programming_fever',draw_function)

while(1):

    cv2.imshow("color detection by programming_fever",img)
    if (clicked):
   
        #cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 fills entire rectangle 
        cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)

        #Creating text string to display( Color name and RGB values )
        text = getColorName(r,g,b) + ' R='+ str(r) +  ' G='+ str(g) +  ' B='+ str(b)
        
        #cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
        cv2.putText(img, text,(50,50),2,0.8,(255,255,255),2,cv2.LINE_AA)

        #For very light colours we will display text in black colour
        if(r+g+b>=600):
            cv2.putText(img, text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
            
        clicked=False

    if cv2.waitKey(20) & 0xFF ==27:
        break
    
cv2.destroyAllWindows()















# import numpy as np
# import sys
# import cv2 as cv
# def show_wait_destroy(winname, img):
#     cv.imshow(winname, img)
#     cv.moveWindow(winname, 500, 0)
#     cv.waitKey(0)
#     cv.destroyWindow(winname)
# def main(argv):
#     # [load_image]
#     # Check number of arguments
#     if len(argv) < 1:
#         print ('Not enough parameters')
#         print ('Usage:\nmorph_lines_detection.py < path_to_image >')
#         return -1
#     # Load the image
#     src = cv.imread(argv[0], cv.IMREAD_COLOR)
#     # Check if image is loaded fine
#     if src is None:
#         print ('Error opening image: ' + argv[0])
#         return -1
#     # Show source image
#     cv.imshow("src", src)
#     # [load_image]
#     # [gray]
#     # Transform source image to gray if it is not already
#     if len(src.shape) != 2:
#         gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
#     else:
#         gray = src
#     # Show gray image
#     show_wait_destroy("gray", gray)
#     # [gray]
#     # [bin]
#     # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
#     gray = cv.bitwise_not(gray)
#     bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
#                                 cv.THRESH_BINARY, 15, -2)
#     # Show binary image
#     show_wait_destroy("binary", bw)
#     # [bin]
#     # [init]
#     # Create the images that will use to extract the horizontal and vertical lines
#     horizontal = np.copy(bw)
#     vertical = np.copy(bw)
#     # [init]
#     # [horiz]
#     # Specify size on horizontal axis
#     cols = horizontal.shape[1]
#     horizontal_size = cols // 30
#     # Create structure element for extracting horizontal lines through morphology operations
#     horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
#     # Apply morphology operations
#     horizontal = cv.erode(horizontal, horizontalStructure)
#     horizontal = cv.dilate(horizontal, horizontalStructure)
#     # Show extracted horizontal lines
#     show_wait_destroy("horizontal", horizontal)
#     # [horiz]
#     # [vert]
#     # Specify size on vertical axis
#     rows = vertical.shape[0]
#     verticalsize = rows // 30
#     # Create structure element for extracting vertical lines through morphology operations
#     verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
#     # Apply morphology operations
#     vertical = cv.erode(vertical, verticalStructure)
#     vertical = cv.dilate(vertical, verticalStructure)
#     # Show extracted vertical lines
#     show_wait_destroy("vertical", vertical)
#     # [vert]
#     # [smooth]
#     # Inverse vertical image
#     vertical = cv.bitwise_not(vertical)
#     show_wait_destroy("vertical_bit", vertical)
#     '''
#     Extract edges and smooth image according to the logic
#     1. extract edges
#     2. dilate(edges)
#     3. src.copyTo(smooth)
#     4. blur smooth img
#     5. smooth.copyTo(src, edges)
#     '''
#     # Step 1
#     edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
#                                 cv.THRESH_BINARY, 3, -2)
#     show_wait_destroy("edges", edges)
#     # Step 2
#     kernel = np.ones((2, 2), np.uint8)
#     edges = cv.dilate(edges, kernel)
#     show_wait_destroy("dilate", edges)
#     # Step 3
#     smooth = np.copy(vertical)
#     # Step 4
#     smooth = cv.blur(smooth, (2, 2))
#     # Step 5
#     (rows, cols) = np.where(edges != 0)
#     vertical[rows, cols] = smooth[rows, cols]
#     # Show final result
#     show_wait_destroy("smooth - final", vertical)
#     # [smooth]
#     return 0
# if __name__ == "__main__":
#     main(sys.argv[1:])



# import sys                                          # System bindings
# import cv2                                          # OpenCV bindings
# import numpy as np

# def rescaleFrame(frame, scale=1):
#     # Images, Videos and Live Video
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)

#     dimensions = (width,height)

#     return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# img = cv2.imread('images/Realsense.jpg')
# color = rescaleFrame(img)
# cv2.imshow('Original',color)
# cv2.moveWindow("Original",0,0)
# print(color.shape)


# # height,width,channels = color.shape

# b,g,r = cv2.split(color)


# blank = np.zeros(color.shape[:2],dtype='uint8')
# blue = cv2.merge([b,blank,blank])
# green = cv2.merge([blank,g,blank])
# red = cv2.merge([blank,blank,r])
# cv2.imshow('Blue',blue)
# cv2.imshow('Green',green)
# cv2.imshow('Red',red)
# # rgb_split = np.empty([height,width*3,3],'uint8')
# # rgb_split[:, 0:width] = cv2.merge([b,b,b])
# # rgb_split[:, width:width*2] = cv2.merge([g,g,g])
# # rgb_split[:, width*2:width*3] = cv2.merge([r,r,r])

# # cv2.imshow("Channels",rgb_split)
# # cv2.moveWindow("Channels",0,height)

# # hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
# # h,s,v = cv2.split(hsv)
# # hsv_split = np.concatenate((h,s,v),axis=1)

# #cv2.imshow("Split HSV",hsv_split)
# cv2.waitKey(0)
# cv2.destroyAllWindows()