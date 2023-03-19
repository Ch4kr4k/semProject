import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from PIL import Image
import numpy as np
model_path = '/media/zera/DATA/Project/semProject4/exported_model/gesture_recognizer.task'
import time

import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with GestureRecognizer.create_from_options(options) as recognizer:
    #cap = cv2.VideoCapture('/home/zera/Documents/vid.mp4')
    cap = cv2.VideoCapture(0)
    

    while cap.isOpened():
       
       
       success, frame= cap.read()

       frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)      
       
       
       #pil_img = Image.fromarray(frame)

       image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

       
       
       #mp_image = mp.Image(format=mp.ImageFormat.SRGB, data=image)
       #current_time = int(time.time() * 1000000)
       #frame_timestamp_ms = mp.Timestamp(current_time)
       
       

       recognizer.recognize_async(image,int(frame_timestamp_ms))
       
       

       #top_gesture = recognition_result.gestures[0][0]

       #print(top_gesture.category_name)


       cv2.imshow('MediaPipe Face Mesh', cv2.flip(frame, 1))
       if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()
