# Face-Mask-Detection

In this project, I've used Transfer Learning approach to build a Face Mask Detector. The initial motivation behind this project was to contribute something in these pressing times of a pandemic. Studies have shown that wearing masks can reduce the risk of contamination by almost 90%. Anyway, coming back to this project, I've used keras-vggface pre trained model, fine tuned it and trained it on my custom dataset. It should absolute amazing reults by giving upto 100% accuracy on cross validation set (Why it waas so, will come back in a while). After saving the model, I used state-of-the-art face detection algorithm, MTCNN to detect faces in both still and moving images. After finding out the Region Of Interest (ROI) of the face, I used only this portion and fed it to my saved model to predict the presence of mask. The main target audience for this project is public places which can automate their regulatory measures of admission by recognizing people without mask.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Firstly clone this repo in your local system by the following command.
```
git clone https://github.com/IP1102/Face-Mask-Detection.git
```
Install the dependencies in your environment by the following command. Note: Python 3.4+ is must.
```
pip install -r requirements.txt
```

### Testing the project
To test it, simply run the face_mask_detection(webcam).py file as follows.
```
python face_mask_detection(webcam).py
```
## Dataset
The dataset that I used in this project can be found [here](https://app.monstercampaigns.com/c/tortsem7qkvyuxc4cyfi). This dataset contains 1376 images of 2 classes - with_mask (690) and without_mask(686). The dataset is created by Prajna Bhandary. She took pictures of naked faces and applied a digital mask on top of the face to create the masked faces dataset. Brilliant! But this lead to that previously mention 100% accuracy on validation set, because each picture had the same blue surgical mask. So applying other masks in this dataset will increase the model performance. 

## File Descriptions
* [face_detect(HaarCascade).py](https://github.com/IP1102/Face-Mask-Detection/blob/master/face_detect(HaarCascade).py) - Detects faces in an image or a frame by the pre defined cascades by OpenCV. But failed to detect face ROI when mask was worn. I've added it if anyone wants to play around it. Just have the data folder in this project directory and copy the path of the cascade model that you want to load. It also offers few facial reognition pre trained models. 
* [face_mask_detect(static).py](https://github.com/IP1102/Face-Mask-Detection/blob/master/face_mask_detect(static).py) - Detects whether an image file contains a face with mask. It's not very useful, becaus the core objective of this project is to detect mask in real time. 
* [face_mask_detect(webcam).py](https://github.com/IP1102/Face-Mask-Detection/blob/master/face_mask_detect(webcam).py) - This is the actual code. Running this file will open the 1st webcam of the laptop/desktop and predict whether the frame has a face and if the face has mask on or not. 
* [image_preprocess.py](https://github.com/IP1102/Face-Mask-Detection/blob/master/image_preprocess.py) - Preprocess the image or frame before making the predictions.
* [face_mask_train.ipynb](https://github.com/IP1102/Face-Mask-Detection/blob/master/face_mask_train.ipynb) - This is the main training notebook. Here, you can find the other pre-trained models also tried and tested but were overfitting, so the best model was found with keras_vggface model. I've tried to keep it as simple as possible so that one can easily retrain with their own data or want to just play around with the layers. 
* [models/face_mask_vggface_vgg16.h5](https://github.com/IP1102/Face-Mask-Detection/blob/master/models/face_mask_vgg16.h5) - The vggface model built on top of VGG16 which I finetuned and trained on the custom dataset. You can directly use this model to make the predictions in your code. 

## Built With 
* [MTCNN](https://github.com/ipazc/mtcnn) - State-Of-The-Art face detection library. 
* [keras-vggface](https://github.com/rcmalli/keras-vggface) - Oxford VGGFace Implementation using Keras Functional Framework v2+.

## Authors 
[Indranil Palit](https://github.com/IP1102).

## Acknowledgements
* [Prajna Bhandary](https://www.linkedin.com/feed/update/urn%3Ali%3Aactivity%3A6655711815361761280/) for the amazing dataset.
* This project was highly inspired from this PyImageSearch [article](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/).
