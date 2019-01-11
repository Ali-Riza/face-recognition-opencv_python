import cv2
import numpy as np


faceDetect = cv2.CascadeClassifier("C:/AliRiza_Scripts_Python/Face_Recognition/haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0);
rec = cv2.face.LBPHFaceRecognizer_create();
rec.read("C:/AliRiza_Scripts_Python/Face_Recognition/recognizer/trainningData.yml")
id = 0
person = ("Hallo")
font = cv2.FONT_HERSHEY_SIMPLEX

url = "http://192.168.2.110:8080/shot.jpg"

while(True):
    ret,img = cam.read();
    imgResp = urllib.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()),dtype = np.uint8)
    img = cv2.imdecode(imgNp,-1)
    cv2.imshow("test",img)
    if ord("q")==cv2.waitKey(10):
        exit()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),2)
        id,conf = rec.predict(gray[y:y+h,   x:x+w])
        if(id==3):
            person =("Ali Riza")
    
        if(id==2):
            person = ("Akif")

        if(id==4):
            person = ("Ahmet Bagci")

        cv2.putText(img, "Name: " + str(person), (x,y+h), font, 1, (0,0,255), 2, cv2.LINE_AA)
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord("q")):
        break;
cam.release()
cv2.destroyAllWindows
