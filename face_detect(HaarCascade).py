import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('Cascades\data\haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while (True):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] # Region of interest in Gray Scale
        roi_color = frame[y:y+h, x:x+w]

        cv2.imwrite('temp_face2.png',roi_color) #Temporarily save the face
        
        #Build the rectangle around the face
        color = (0,0,255)
        stroke = 2
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, stroke)


    #Disply
    cv2.imshow('WebCam View',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


#Close and Destroy
cap.release()
cv2.destroyAllWindows()
