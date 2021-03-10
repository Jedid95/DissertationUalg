'''
    Pose Estimation with KNX interface
    Documentation about code in reference.txt file
    Author: Jedid Santos
    E-mail: jedid.santos@gmail.com
    Date: November 15th, 2020
    Update: March 09th, 2021
    Masters Engineering University of Algarve
'''

#Import for pose estimation
from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2
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

#Import for KNX
import asyncio
import sys
from xknx import XKNX
from xknx.devices import Switch

#Import for KNX data
import pandas as pd
import os

#Import for voice speak
import pyttsx3


#Init voice speak
voice = pyttsx3.init()

#Loading files for KNX software
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
knxFile = os.path.join(THIS_FOLDER, 'enderecos.csv')


#Loading files for pose estimation
ctx = mx.cpu()
detector_name = "ssd_512_mobilenet1.0_coco"
detector = get_model(detector_name, pretrained=True, ctx=ctx)

detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
detector.hybridize()

estimators = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)
estimators.hybridize()

cap = cv2.VideoCapture(0)
time.sleep(1)



# Just a place holder. Actual value calculated after 10 frames.
fps = 10.0
t = cv2.getTickCount()
count = 0

# ============= Format Keypoints ===============================================================================================
# # Nose - 0 | # Right Eye - 1 | # Left Eye - 2 | # Right Ear - 3 | # Left Ear - 4 | # Right Shoulder - 5 | # Left Shoulder - 6
# # Right Elblow - 7 | # Left Elbow - 8 | # Left Wrist - 9 | # Right Wrist - 10 | # Right Hip - 11 | # Left Hip - 12
# # Right Knee - 13 | # Left Knee - 14 | # Right Ankle - 15 | # Left Ankle - 16
# ==============================================================================================================================

#Function to activate a Switch
async def activateSwitch():
    xknx = XKNX()
    await xknx.start()
    switch = Switch(xknx, name="swith", group_address="0/1/0")
    await switch.set_on()
    await asyncio.sleep(2)
    await xknx.stop()

#Function to disable a Switch
async def disableSwitch():
    xknx = XKNX()
    await xknx.start()
    switch = Switch(xknx, name="swith", group_address="0/1/0")
    await switch.set_off()
    await asyncio.sleep(2)
    await xknx.stop()

    
#Function for check position and determine the movement performed
def checkPosition(points,img):
    RightShoulder = points[0,5] #Ombro direito
    RightWrist = points[0,10] #Pulso direito
    LeftShoulder = points[0,6] #Ombro Esquerdo
    LeftWrist = points[0,9] #Pulso Esquerdo


    x_RightShoulder, y_RightShoulder = RightShoulder
    x_RightWrist, y_RightWrist = RightWrist
    x_LeftShoulder, y_LeftShoulder = LeftShoulder
    x_LeftWrist, y_LeftWrist = LeftWrist


    if y_RightWrist < y_RightShoulder and y_LeftWrist < y_LeftShoulder:
        action="Both Arms Up"
        #asyncio.run(activateSwitch())

    elif y_RightWrist < y_RightShoulder:
        action = "Right Arm Up"
        #asyncio.run(activateSwitch())
        return 1
    
    elif y_LeftWrist < y_LeftShoulder:
        action = "Left Arm Up"
        #asyncio.run(disableSwitch())
        return 0

    else: 
        action = "Arms Down"

    cv2.putText(img, action, (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                cv2.LINE_AA)



#Function for read KNX file
def readFile(opcao):

    if opcao == 1:
        df = pd.read_csv(knxFile,encoding = "ISO-8859-1", sep=';')
        print(df)
        voice.say("All file")
        voice.runAndWait()
        menu()
    elif opcao == 2:
        df = pd.read_csv(knxFile,encoding = "ISO-8859-1", sep=';', usecols=['Sub'])
        print(df)
        voice.say("Name devices")
        voice.runAndWait()
        menu()
    elif opcao == 3:
        df = pd.read_csv(knxFile,encoding = "ISO-8859-1", sep=';', usecols=['Address'])
        print(df)
        voice.say("Address devices")
        voice.runAndWait()
        menu()    

#function for main menu
def menu():
    opcao=int(input('''
                        Escolha uma opção:
                        1 - Mostrar todo ficheiro
                        2 - Mostrar nomes dos dispositivos
                        3 - Mostrar endereços dos dispositivos
                        4 - Pose estimation
                        5 - Fechar Menu
                        Escolha:  '''))
    if opcao == 1:
        print("Apresentar todo ficheiro")
        readFile(1)
    elif opcao == 2:
        print("Apresentar nomes dos dispositivos")
        readFile(2)
    elif opcao == 3:
        print("Apresentar endereços dos dispositivos")
        readFile(3)
    elif opcao == 4:
        print("Rodar Pose Estimation")
        poseEstimation()
    elif opcao == 5:
        exit()
    else:
        print("Este número não está nas alternativas, tente novamente")
        menu()


def poseEstimation():
    start = True
    while(True):
        ret, frame = cap.read()
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
            flag = checkPosition(pred_coords,img)
            #if flag == 1:
                #asyncio.run(activateSwitch())

            #elif flag == 0:
                #asyncio.run(disableSwitch())
            
        #size = img.shape[0:2]
        #cv2.putText(img, "fps: {}".format(fps), (25, size[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 3)
        cv_plot_image(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

menu()




