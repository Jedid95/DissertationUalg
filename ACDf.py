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

#import face recognizer
from PIL import Image,ImageTk

#Loading files face detect/recognizer
faceFile = "models/haarcascade_frontalface_default.xml"
datasetFace = "C:/Users/User/Desktop/dataset/"
trainFaceFile = "models/trained_face.yml"
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainFaceFile)
faceCascade = cv2.CascadeClassifier(faceFile)
font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['None','Jedid']

#Loading files for pose estimation
ctx = mx.cpu()
detector_name = "ssd_512_mobilenet1.0_coco"
detector = get_model(detector_name, pretrained=True, ctx=ctx)

detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
detector.hybridize()

estimators = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)
estimators.hybridize()




#Function for initial menu
def menu():
    opcao=int(input('''
                    Welcome ACDf:
                    1 - Start
                    2 - Train Face
                    3 - Exit
                    Escolha:  '''))
    
    if opcao==1:
        poseEstimation()
    if opcao==2:
        takeimgs()
        trainimgs()
    if opcao==3:
        exit()
    else:
        menu()


#Function for take imgs to train Face Recognizer
def takeimgs():
    ID = input('ID: ')
    name = input('Name: ')
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(faceFile)
    sampleNum = 0
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # incrementing sample number
            sampleNum = sampleNum + 1
            # saving the captured face in the dataset folder
            cv2.imwrite(datasetFace + name + "." + ID + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
            cv2.imshow('Train', img)
            # wait for 100 miliseconds
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # break if the sample number is morethan 100
        elif sampleNum > 100:
            break
    cam.release()
    cv2.destroyAllWindows()
    print("Images Saved  : " + ID + " Name : " + name)
    print("Train images......")

#Function for train imgs Face Recognizer
def trainimgs():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    global detectorFace
    detectorFace = cv2.CascadeClassifier(faceFile)
    try:
        global faces,Id
        faces, Id = getImagesAndLabels(datasetFace)
    except Exception as e:
        print('please make "dataset" folder & put Images')

    recognizer.train(faces, np.array(Id))
    try:
        recognizer.save(trainFaceFile)
    except Exception as e:
        print('Please make "model" folder')
    print("Model Trained")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image

        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces = detectorFace.detectMultiScale(imageNp)
        # If a face is there then append that in the list as well as Id of it
        for (x, y, w, h) in faces:
            faceSamples.append(imageNp[y:y + h, x:x + w])
            Ids.append(Id)
    return faceSamples, Ids

def faceRecognizer(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)

#Function for detect body and apply pose detect (Skeleton-based Model)
def poseEstimation():
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    # Just a place holder. Actual value calculated after 10 frames.
    fps = 10.0
    t = cv2.getTickCount()
    count = 0
    start = True
    while(True):
        if count==0:
            t = cv2.getTickCount()
        ret, frame = cap.read()
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

        x, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=350)
        x = x.as_in_context(ctx)
        class_IDs, scores, bounding_boxs = detector(x)

        pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs, output_shape=(128, 96), ctx=ctx)

        if start:
            img = frame
            start = False
            
        #body detected
        if len(upscale_bbox) > 0:
            faceRecognizer(frame)
            predicted_heatmap = estimators(pose_input)
            pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
            img = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores, box_thresh=0.5, keypoint_thresh=0.2)

            #poseRecognition
            #flag = poseRecognition(pred_coords,img)
            #if flag == 1:
                #asyncio.run(activateSwitch())

            #elif flag == 0:
                #asyncio.run(disableSwitch())
        else:
            print('No body')
            
        size = img.shape[0:2]
        cv2.putText(img, "fps: {}".format(fps), (10, size[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        cv_plot_image(img)

        
        # calculate fps at an interval of 10 frames
        count = count + 1
        if (count == 10):
            t = (cv2.getTickCount() - t)/cv2.getTickFrequency()
            fps = 10.0/t
            count = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

#function for recognition movements
#def poseRecognition(points,img):

while True:
    menu()