from __future__ import division
import argparse
import time
import logging
import os
import math
import tqdm
import cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt

import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints

ctx = mx.cpu()
detector_name = "ssd_512_mobilenet1.0_coco"
detector = get_model(detector_name, pretrained=True, ctx=ctx)

detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
detector.hybridize()

estimators = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)
estimators.hybridize()

cap = cv2.VideoCapture(0)
time.sleep(1)

start = True

# Just a place holder. Actual value calculated after 10 frames.
fps = 10.0
t = cv2.getTickCount()
count = 0

# ============= Format Keypoints =============
# # Nose - 0 | # Right Eye - 1 | # Left Eye - 2 | # Right Ear - 3 | # Left Ear - 4 | # Right Shoulder - 5 | # Left Shoulder - 6
# # Right Elblow - 7 | # Left Elbow - 8 | # Left Wrist - 9 | # Right Wrist - 10 | # Right Hip - 11 | # Left Hip - 12
# # Right Knee - 13 | # Left Knee - 14 | # Right Ankle - 15 | # Left Ankle - 16
# ==============================================================================================================================

#Function for check position and determine the movement performed
def checkPosition(points,img):
    flag = False
    RightShoulder = points[0,5] #Ombro direito
    RightWrist = points[0,10] #Pulso direito
    LeftShoulder = points[0,6] #Ombro Esquerdo
    LeftWrist = points[0,9] #Pulso Esquerdo


    x_RightShoulder, y_RightShoulder = RightShoulder
    x_RightWrist, y_RightWrist = RightWrist
    x_LeftShoulder, y_LeftShoulder = LeftShoulder
    x_LeftWrist, y_LeftWrist = LeftWrist


    if y_RightWrist < y_RightShoulder and y_LeftWrist < y_LeftShoulder:
        #action="Both Arms Up"
        flag = True
    elif y_RightWrist < y_RightShoulder:
        action = "Right Arm Up"
        
    
    elif y_LeftWrist < y_LeftShoulder:
        action = "Left Arm Up"
        #flag = True

    else: action = "Arms Down"
   
    #size = img.shape[0:2]
    #cv2.putText(img, action, (25, size[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,cv2.LINE_AA)
    if flag == True:
        cv2.imwrite('image5.jpg', img)
        flag = False

while(True):
    if count==0:
      t = cv2.getTickCount()

    #ret, frame = cap.read()
    frame = cv2.imread( 'teste.jpg' )
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

    x, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=350)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs, output_shape=(128, 96), ctx=ctx)

    if start:
        img = frame
        start = False

    if len(upscale_bbox) > 0:
        predicted_heatmap = estimators(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

        img = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores, box_thresh=0.5, keypoint_thresh=0.2)
        checkPosition(pred_coords,img)
    #print(pred_coords[0,5])
    size = img.shape[0:2]
    #cv2.putText(img, "fps: {}".format(fps), (25, size[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)
    cv_plot_image(img)
    
    count = count + 1
    # calculate fps at an interval of 10 frames
    if (count == 10):
        t = (cv2.getTickCount() - t)/cv2.getTickFrequency()
        fps = 10.0/t
        count = 0
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
