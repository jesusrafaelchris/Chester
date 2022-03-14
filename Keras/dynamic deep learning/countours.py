import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    ret,frame = cap.read()

    frame = cv2.GaussianBlur(frame, (7, 7), 1.41) # put camera through gaussian filter to blur it
    frame = cv2.medianBlur(frame,5)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # turn the blurred image to greyscale

    ret, thresh = cv2.threshold(grey, 127, 255, 1)

    # Extract Contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #frame = cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if(cv2.contourArea(c) > 2000) and (cv2.contourArea(c) < 15000) :
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #frame = cv2.drawContours(frame,[box],0,(0,0,255),2)
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            #hull = cv2.convexHull(c)
            #frame = cv2.drawContours(frame,[hull],0,(255,0,0),2)
            #epsilon = 0.1*cv2.arcLength(c,True)
            #approx = cv2.approxPolyDP(c,epsilon,True)
            #frame = cv2.drawContours(frame,[approx],0,(100,100,100),2)
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)
            center = (int((x1+x2)/2), int((y1+y2)/2))
            cv2.circle(frame, center, 8, (0,0,255), -1)
            roi = frame[y:y+h, x:x+w]
            for i in range(10):
                cv2.imwrite('/Users/christiangrinling/Desktop/images/img{}.jpg'.format(i), roi)


    cv2.imshow('window', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
