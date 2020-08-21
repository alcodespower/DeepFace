import cv2 as cv

img = cv.imread('mahuateng.jpg')
x,y,w,h = 550,210,230,230

img = cv.rectangle(img,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)
cv.imshow('result image',img)
cv.waitKey(0)
cv.destroyAllWindows()