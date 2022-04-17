from cv2 import LINE_4, LINE_AA
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
img=cv2.imread('happy.jpg')
plt.imshow(img)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
cv2.waitKey()
prediction=DeepFace.analyze(img)
# print(prediction['dominant_emotion'])
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
# detectMultiScale(image, rejectLevels, levelWeights)
# here rejectlevels means how much the image size is reduced with each scale
# levelWeight show many neighbors each candidate rectangle should have to retain it.
# detectMultiScale function is used to detect the faces.
faces=faceCascade.detectMultiScale(gray,1.2,5)
for(x,y,w,h) in faces:
    # (x,y),((x+w),(y+h)) are two points.
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# cv2.imshow('emotiondetect',img)
# cv2.waitKey()
font=cv2.FONT_HERSHEY_PLAIN
cv2.putText(img,prediction['dominant_emotion'],(0,50),font,4,(255,255,0),4,LINE_4)
cv2.imshow('emotiondetect',img)
cv2.waitKey()


