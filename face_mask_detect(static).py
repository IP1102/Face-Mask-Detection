import cv2
import numpy as np
from keras.models import load_model
from image_preprocess import preprocess

model = load_model('models/face_mask_vggface_vgg16.h5')

image = cv2.imread('Test Images/temp_face_mask.png')
image = preprocess(image)

pred_temp = model.predict(image)
pred = np.argmax(pred_temp, axis = 1)

if (pred==0):
    print("Mask Detected")
else:
    print("No Mask Detected")
