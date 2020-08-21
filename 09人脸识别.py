import os,cv2,sys
from PIL import Image
import numpy as np
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
img2 = cv2.imread('./data/jm/1.pgm')
img = cv2.imread('./data/jm/check/10.pgm')
face_detector = cv2.CascadeClassifier(
        'D:/A_Application_All/Application/Anaconda3/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(gray)
for x, y, w, h in faces: 
    cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    id,confidence = recognizer.predict(gray[y:y+h,x:x+w])
    print('标签ID；',id,'置信评分：',confidence)
cv2.imshow('result', img)
cv2.imshow('result2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
