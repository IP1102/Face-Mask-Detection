import cv2
import numpy as np

def preprocess(im):
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (224,224))
    im = im / 255
    im = np.expand_dims(im,axis=0)
    return im