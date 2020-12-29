import numpy as np
import cv2

# Load the image
img1 = cv2.imread('rightUp6_estimation.png')
img2 = cv2.imread('rightUp5_estimation.png')
img3 = cv2.imread('Arm_estimation.png')
img1 = cv2.resize(img1, (512, 350))
img2 = cv2.resize(img2, (512, 350))
img3 = cv2.resize(img3, (512, 350)) 
# Horizontally concatenate the 2 images
img4 = cv2.hconcat([img1,img2,img3])
 
# Display the concatenated image
cv2.imshow('a2',img4)
cv2.waitKey(0)