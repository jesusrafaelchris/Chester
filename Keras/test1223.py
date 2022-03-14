import cv2
import numpy as np
from collections import defaultdict
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import os, sys
import glob

def segmentLines(lines, delta): #Split all lines into vertical or horizontal
    h_lines = []
    v_lines = []
    for i in range(len(lines)):
        if abs(lines[i][0][1])<delta or abs(lines[i][0][1]-np.pi)<delta:
            h_lines.append(lines[i])
        else:
            v_lines.append(lines[i])
    return h_lines, v_lines

def findStrongLines(lines,rho_dif,theta_dif,num): #find the lines that have the most evidence to support their existence
    strongLines = []
    strongLines.append(lines[0])
    for i in range(1, len(lines)):
        if len(strongLines) >= num:
            break
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        for z in range(len(strongLines)):
            if (abs(rho - strongLines[z][0][0])<rho_dif and abs(theta - strongLines[z][0][1])<theta_dif):
                break
            elif z == len(strongLines)-1:
                strongLines.append(lines[i])
    return strongLines

def findIntersections(h_lines,v_lines): #return x and y cords for points of intersection between horizontal and vertical lines
    intersections = []
    for i in range(len(h_lines)):
        for z in range(len(v_lines)):
            rho1 = h_lines[i][0][0]
            theta1 = h_lines[i][0][1]
            rho2 = v_lines[z][0][0]
            theta2 = v_lines[z][0][1]
            x = (np.sin(theta2)*rho1-rho2*np.sin(theta1))/(np.cos(theta1)*np.sin(theta2)-np.cos(theta2)*np.sin(theta1))
            y = (rho2 - x*np.cos(theta2))/np.sin(theta2)
            x = int(round(x))
            y = int(round(y))
            intersections.append([x,y])
    return intersections

def resize(ROI):
    root_dir = "/Users/christiangrinling/Desktop/Keras/extracted"
    for filename in glob.iglob(root_dir + '**/*.jpg'):
        print(filename)
        im = Image.open(filename)
        imResize = im.resize((224,224), Image.ANTIALIAS)
        imResize.save(filename , 'JPEG', quality=90)
        return ROI


def main():

    windowName = "HoughLineDetector"
    cv2.namedWindow(windowName)
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    lines=[]
    intersections = []

    while ret:

        ret, frame = cap.read()

        frame = cv2.GaussianBlur(frame, (7, 7), 1.41) # put camera through gaussian filter to blur it

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # turn the blurred image to greyscale

        edges = cv2.Canny(grey, 25, 75) # run the greyscale footage through canny detector

        '''corners = cv2.goodFeaturesToTrack(grey,4,0.4,50) # find the corners of greyscale image
        if corners is not None:
            for corner in corners:
                x,y = corner.flatten()
                cv2.circle(frame,(x,y),5,(36,255,12),-1)'''

        lines = cv2.HoughLines(edges, 1, np.pi/180, 150) #run canny "edges" through hough line detector

        if lines is not None:
            h_lines, v_lines = segmentLines(lines,np.pi/4)

            #Collect only the first 100 lines.
            #Arguments are lines to sort, pixel rho must differ by and degrees theta must differ by and number of lines to find

            if h_lines is not None:
                if len(h_lines)>0:
                    h_lines = findStrongLines(h_lines,30,20,9)
                for i in range(0, len(h_lines)): #Draw horizontal in red
                    #print(h_lines[i])
                    rho = h_lines[i][0][0]
                    theta = h_lines[i][0][1]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if v_lines is not None:
                if len(v_lines)>0:
                    v_lines = findStrongLines(v_lines,30,20,9)
                for i in range(0, len(v_lines)): #Draw horizontal in blue
                    rho = v_lines[i][0][0]
                    theta = v_lines[i][0][1]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))

                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if h_lines is not None and v_lines is not None:
                intersections = findIntersections(h_lines,v_lines) #if there are both horizontal and vertical lines present, find intersections
                for i in range(len(intersections)):
                    x,y = intersections[i]
                    cv2.circle(frame,(x,y),5,(36,255,12),-1)

        cv2.imshow(windowName, frame) #show image result

        if len(intersections) == 81: #stop searching if 81 intersetions found
            global img
            img = frame
            break

        if cv2.waitKey(1) == 27: # exit on ESC
            break

    cv2.destroyAllWindows()
    cap.release()

    '''Sort intersections into 9x9 array in their relative positions on the board'''


    #create 9x9 array to append to
    cords = []
    for i in range(9):
        cords.append([])
        for j in range(9):
            cords[i].append([])

    #calculate top left cord
    smallestVal = 99999999
    smallestPos = 0
    for i in range(len(intersections)):
        value = pow(intersections[i][0],2)+pow(intersections[i][1],2)
        if value < smallestVal:
            smallestPos = i
            smallestVal = value
    cords[0][0] = intersections[smallestPos]

    #calculate top row
    temp = []
    '''range to check for cords in the same column/row'''
    verDisp = 20
    while len(temp) < 8:
        verDisp = verDisp + 5
        for i in range(len(intersections)):
            if abs(cords[0][0][1]-intersections[i][1])-verDisp < 0:
                temp.append(intersections[i])

    temp=sorted(temp)
    del temp[0] #remove first value which is the top row, and thus already stored
    while len(temp) > 8:#make sure only 8 values in temp, so a total of 9 will be added to array
        del temp[8]

    #Append top row to 9x9
    for i in range(len(temp)):
        cords[0][i+1] = temp[i]

    #calculate columns
    for z in range(9):
        temp = []
        '''range to check for cords in the same column/row'''
        horDisp = 20
        while len(temp) < 8:
            horDisp = horDisp + 5
            for i in range(len(intersections)):
                if abs(cords[0][z][0]-intersections[i][0])-horDisp < 0:
                    temp.append(intersections[i])

        temp=sorted(temp,key=lambda x: x[1]) #sort, but by the y cord not x this time
        del temp[0]
        while len(temp) >8: #make sure only 8 values in temp, so a total of 9 will be added to array
            del temp[8]
        #print(z)
        #print(temp)

        #Append column to 9x9
        for i in range(len(temp)):
            cords[i+1][z] = temp[i]

    for i in range(9):
        print(cords[i])

    '''#display top left cords
    while True:
        cv2.circle(img,(cords[0][0][0],cords[0][0][1]),5,(200,0,200),-1)
        cv2.imshow(windowName, img) #show image result
        if cv2.waitKey(1) == 27: # exit on ESC
                break

    cv2.destroyAllWindows()
    cap.release()'''

    '''Cords have been collected and now must be looped through to calculate the top left and bottom right
        for each sqaure, so the piece in that sqaure can be predicted'''

            ####extract each image using the topLeft and bottomRight values, that will provide a square
            ####containing the piece.


    for y in range(8):      #loop through items in the column from position 0 to 7 ie 8 sqaures
        for x in range(8): #loop through items in the row from position 0 to 7 ie 8 squares
            topLeft = cords[y][x]
            bottomRight = cords[y+1][x+1]
            print("topleft",topLeft)
            print("bottomright",bottomRight)
            x1 = topLeft[0]
            y1 = topLeft[1]
            x2 = bottomRight[0]
            y2 = bottomRight[1]
            print("x1 is " ,x1, "y1 is ", y1,"\n" , "x2 is ", x2, "y2 is ",y2)
            #cords1 = np.array([[x1],[x2],[y1],[y2]])
            cords1 = np.array([x1,x2,y1,y2]) #Puts the coordinates into an array
            print('cords',cords[y][x])
    cv2.imshow('tt', img_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
