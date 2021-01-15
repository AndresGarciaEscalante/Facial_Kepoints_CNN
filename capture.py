import cv2, time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from models import Net

# Load the model 
net = Net()
net.load_state_dict(torch.load('saved_models/keypoints_model_1.pt'))

# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# Create an object, Zero for external camera
video = cv2.VideoCapture(0)
a = 0
while True:
    a= a+1

    check, frame = video.read()

    #print(check)
    #print(frame) # Representing the image 

    # 6. Converting to grayscale
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # run the detector
    # the output here is an array of detections; the corners of each detection box
    # if necessary, modify these parameters until you successfully identify every face in a given image
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    # loop over the detected faces, mark the image where each face is found
    for (x,y,w,h) in faces:
        # draw a rectangle around each detected face
        # you may also need to change the width of the rectangle drawn depending on image resolution
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3) 

    # Show the frame!
    cv2.imshow("Capturing",frame)

    key = cv2.waitKey(1)

    # End the streaming if q is pressed
    if key == ord('q'):
        break
print(a)
video.release()
cv2.destroyAllWindows