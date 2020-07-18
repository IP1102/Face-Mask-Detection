import numpy as np
import cv2
from keras.models import load_model
from image_preprocess import preprocess

from mtcnn.mtcnn import MTCNN

detector = MTCNN()

# face_cascade = cv2.CascadeClassifier('Cascades\data\haarcascade_frontalface_alt2.xml')
model = load_model('models/face_mask_vggface_vgg16.h5')
cap = cv2.VideoCapture(0)

while (True):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    #Detect Face using MTNCC
    faces = detector.detect_faces(image)

    if faces!=[]:
        color = (255,0,0)
        stroke = 2
        x = faces[0]['box'][0]
        y = faces[0]['box'][1]
        w = faces[0]['box'][2]
        h = faces[0]['box'][3]
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, stroke)

        #Detect whether mask is present in the region of interest (Face)
        roi = frame[y:y+h,x:x+w]
        roi = preprocess(roi) 
        pred_temp = model.predict(roi)
        pred = np.argmax(pred_temp,axis=1)

        print(pred)
        if (pred==0):
            cv2.putText(frame,
            "MASK",(100,100),cv2.FONT_HERSHEY_SIMPLEX,
            1, (0,255,255),2,
            cv2.LINE_4)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), stroke)
        else:
            cv2.putText(frame,
            "NO MASK",(100,100),cv2.FONT_HERSHEY_SIMPLEX,
            1, (0,128,255),2,
            cv2.LINE_4)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), stroke)

    
    #Disply
    cv2.imshow('Face Mask Detector',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


#Close and Destroy
cap.release()
cv2.destroyAllWindows()
