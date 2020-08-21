import os,cv2,sys
from PIL import Image
import numpy as np
def getImageAndLabels(path):
    faceSamples = []
    ids = []
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    face_detector = cv2.CascadeClassifier('D:/A_Application_All/Application/Anaconda3/opencv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml')



    for imagePath in imagePaths:
        PIL_image = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_image,'uint8')
        faces = face_detector.detectMultiScale(img_numpy)
        id = int(os.path.split(imagePath)[1].split('.')[0])
        for x,y,w,h in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)


    print(imagePaths)
    return faceSamples,ids
if __name__ == '__main__':
    path = './data/jm/'
    faces,ids = getImageAndLabels(path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,np.array(ids))
    recognizer.write('trainer/trainer.yml')
