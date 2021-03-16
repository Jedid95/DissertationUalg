import cv2
import numpy as np
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap2 = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)
address = "https://192.168.1.71:8080/video"
cap.open(address)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    _, img2 = cap2.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces2 = face_cascade.detectMultiScale(gray2, 1.1, 4)
    if np.any(faces):
        print('Rosto camera 1')
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    elif np.any(faces2):
        print('Rosto camera 2')
        for (x, y, w, h) in faces2:
            cv2.rectangle(img2, (x, y), (x+w, y+h), (255, 0, 0), 2)
    else:
        print('Sem Rosto')
        
    # Display
    cv2.imshow('img', img)
    cv2.imshow('img2', img2)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()
cap2.release()