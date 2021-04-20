import os
import cv2
import numpy as np

path = "front_face.xml" 
faceCascade = cv2.CascadeClassifier(path)
count = 0
name = input()
name = name.lower()
if name not in os.listdir():
    os.mkdir(name)
else:
    lis = os.listdir(name)
    lis.sort()
    count = int(lis[-1][4:-4])
cap = cv2.VideoCapture(0) # capture video using webcam
cap.set(3,640) # set Width
cap.set(4,480) # set Height
frames = 0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,     
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.imwrite('Known_Faces\' + name + '\img_' + str(count) + '.jpg', roi_color)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        count += 1
        frames += 1
    cv2.imshow('video',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
    if frames >= 1000:
        break
cap.release()
cv2.destroyAllWindows()
