
import tensorflow as tf

import keras.models as md

import cv2

import boto3

from dotenv import load_dotenv

import os

import numpy as np

load_dotenv()

pth= os.getcwd()

model= md.load_model(os.path.join(pth,'models/new_conf_model.h5'))

IMG_SIZE= (299,299)

def VideoConfidence(video_path):  
    key= os.getenv('API_KEY_ID')
    access_key= os.getenv('API_SECRECT_KEY')
    s3=boto3.client('s3',aws_access_key_id=key,aws_secret_access_key=access_key)
    s3.download_file('video-store-hiremeclub',video_path,video_path)
    i=0
    video= cv2.VideoCapture(video_path)
    c=0
    cl=0
    writer= None
    (Width, Height)= (None,None)

    while True:
      video.set(cv2.CAP_PROP_POS_FRAMES, i)
      (taken,frame)= video.read()
      if not taken:
        break
      if Width is None or Height is None:
        (Width,Height)= frame.shape[:2]
      output= frame.copy()
      frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      frame= cv2.resize(frame, IMG_SIZE).astype("float32")
      frame= np.array(frame)
      frame= tf.keras.applications.xception.preprocess_input(frame)
      frame= np.expand_dims(frame,0)
      pred= model.predict(frame)[0]
      ind= np.argmax(pred)

      if ind==1:
        cl+=pred[1]
      c+=1

      if key== ord("q"):
        break
      i+=10

    score= float(cl/c)*100.0
    return score

