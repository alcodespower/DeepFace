import cv2 as cv
def face_detect_demo():
    #灰度
    a,b = img.shape[0:2]
    img_resize= cv.resize(img,(int(b*1),int(a*1)))
    gray = cv.cvtColor(img_resize,cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier('D:/A_Application_All/Application/Anaconda3/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')

    faces = face_detector.detectMultiScale(gray,minSize=(55,55))

    for x,y,w,h in faces:
        print(x,y,w,h)
        cv.rectangle(img_resize,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
    cv.imshow('result',img_resize)
    cv.imwrite('meixi_result.jpg',img_resize)
img = cv.imread('meixi.jpg')
face_detect_demo()
cv.waitKey(0)
cv.destroyAllWindows()
