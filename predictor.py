#!/usr/bin/env python

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import cv2
import numpy as np
import sys
def main():
    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480
    base_options = python.BaseOptions(model_asset_path='rock_paper_siccor.task')
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    #image = 's.jpg'
    image = mp.Image.create_from_file(sys.argv[1])
    recognition_result = recognizer.recognize(image)
    top_gesture = recognition_result.gestures[0][0]
    print(top_gesture.category_name)

if __name__ == '__main__':
    main()