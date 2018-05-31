import numpy as np
import cv2
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

cam=cv2.VideoCapture(0)
while True:

    #img = cv2.imread('kimjongun.jpg')
    ret, img=cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #cv2.putText(img, 'Krisna eyes ' , (x, y), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray,1.2,15)
        nose_rects = nose_cascade.detectMultiScale(roi_gray, 1.1, 15)
        mouth_rects = mouth_cascade.detectMultiScale(roi_gray, 1.3, 15)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        for (ex,ey,ew,eh) in nose_rects:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        for (ex,ey,ew,eh) in mouth_rects:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    #for (x, y, w, h) in mouth_rects:
    #    y = int(y - 0.15 * h)
    #    cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
    #    break

    #for (x, y, w, h) in nose_rects:
    #    y = int(y - 0.15 * h)
    #    cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
    #    break

    cv2.imshow('result',img)
    if cv2.waitKey(10)==ord('q'):
        break


cam.release()
cv2.destroyAllWindows()