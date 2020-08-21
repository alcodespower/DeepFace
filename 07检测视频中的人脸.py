import cv2 as cv
def face_detect_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier(
        'D:/A_Application_All/Application/Anaconda3/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')

    faces = face_detector.detectMultiScale(gray)

    for x, y, w, h in faces:
        print(x, y, w, h)
        cv.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
    cv.imshow('result',img)

cap = cv.VideoCapture('buzuo.mp4')
# cap = cv.VideoCapture(0)
while True:
    flag,frame = cap.read()
    if not flag:
        break
    counter += 1
    face_detect_demo(frame)
    if ord('q') == cv.waitKey(1):
        break
cv.destroyAllWindows()
cap.release()