import cv2
from cv2 import CascadeClassifier
from deepface import DeepFace
# CascadeClassifier is used to load haar cascade files in python
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
try:
    cam=cv2.VideoCapture(0)
except:
        raise IOError("Can't Access WebCam. ")
while True:
    ret,frame=cam.read()
    # here deepface will analyze emotion on each frame
    result=DeepFace.analyze(frame,actions=['emotion'])
    # to draw rectangle convert the frame into gray scale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # detetctMultiScale to detect the position of faces in the image 
    # scale_factor(1.2) specifies the scale by which the size of the image must be reduced and
    # min_neighbours(5) specifies the number of rectangles each rectangle should have.
    face_in_frame=faceCascade.detectMultiScale(gray,1.5,5)
    # here x,y,w,h specifies the position of the image detected by multisacle funcation
    for(x,y,w,h) in face_in_frame:
    # (x,y),((x+w),(y+h)) are two points.
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    font=cv2.FONT_HERSHEY_PLAIN
    cv2.putText(frame,result['dominant_emotion'],(0,50),font,4,(255,255,0),4,cv2.LINE_4)
    cv2.imshow('Video',frame)
    if cv2.waitKey(10) & 0xFF==ord('x'):
        break
cam.release()
cv2.destroyAllWindows()


