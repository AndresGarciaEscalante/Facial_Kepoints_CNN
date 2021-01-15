import cv2, time
import torch
import numpy as np
from models import Net

# Load the model 
net = Net()
net.load_state_dict(torch.load('saved_models/keypoints_model_1.pt'))

# load in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# Create an object, Zero for external camera
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    # run the detector
    # the output here is an array of detections; the corners of each detection box
    # if necessary, modify these parameters until you successfully identify every face in a given image
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)

    # loop over the detected faces, mark the image where each face is found
    for (x,y,w,h) in faces:
        # draw a rectangle around each detected face
        # you may also need to change the width of the rectangle drawn depending on image resolution
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3) 

        #crop the part of the image that the contains the face
        roi_org = frame[y:y+h, x:x+w]

        #change the rgb to gray 
        roi = cv2.cvtColor(roi_org, cv2.COLOR_BGR2GRAY)
    
        ##  Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
        roi = roi/255.0
        
        # Store the dimesion of the original image
        original_shape = roi.shape

        ## Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
        roi = cv2.resize(roi, (224, 224)) 
        
        ## Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
        roi = np.expand_dims(roi, 0)
        roi = np.expand_dims(roi, 0)
        roi = torch.from_numpy(roi)
        
        ## Make facial keypoint predictions using your loaded, trained network 
        ## perform a forward pass to get the predicted facial keypoints
        roi = roi.type(torch.FloatTensor)
        kps = net(roi)

        ## Un-transform the predicted key_pts data
        kps = kps.view(68, -1)
        kps = kps.data
        kps = kps.numpy()

        # undo normalization of keypoints  
        kps = kps * 80.0 + 100
        kps[:, 0] = kps[:, 0] * original_shape[0] / 224
        kps[:, 1] = kps[:, 1] * original_shape[1] / 224

        # Plot the dots in the image
        for (x, y) in zip(kps[:, 0], kps[:, 1]):
            cv2.circle(roi_org, (x, y), 3, (0, 255, 0), -1)

    # Show the frame!
    cv2.imshow("Capturing",frame)

    key = cv2.waitKey(1)

    # End the streaming if q is pressed
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows