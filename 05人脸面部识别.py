import cv2 as cv
def face_detect_demo():
    #灰度
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier('D:/A_Application_All/Application/Anaconda3/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(gray)
    for x,y,w,h in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
    cv.imshow('result',img)
img = cv.imread('mahuateng.jpg')
face_detect_demo()
cv.waitKey(0)
cv.destroyAllWindows()
