import mediapipe as mp
import cv2
import pyautogui as key
from ges_cmd import ges_to_cmd,ges_to_cmd2
import os
import threading
import time

record = ''
frrame_ctr = 0
record_r = ''
record_l=''

model_path = './model/gesture_recognizer.task'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:

def key_cmd(gestures1):

    if gestures1 !='None':
        key.press(ges_to_cmd[gestures1])


def left_cmd(cmd):
    if cmd !='None':
        key.keyDown(ges_to_cmd2[cmd])


def right_cmd(cmd):
    if cmd !='None':
        key.press(ges_to_cmd[cmd],presses=12)


def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):

    top_gesture = result.gestures
    res = None
    res_l = None
    res_r = None

    if len(top_gesture) == 1:

        res = top_gesture[0][0].category_name
        global record

        if record != res:
            key_cmd(res)
            print(res)
            record = res

    elif len(top_gesture) == 2:
        res_r = top_gesture[0][0].category_name  # right -> left in reality
        res_l = top_gesture[1][0].category_name  # left -> right in reality

        global record_r
        if record_r != res_l:
           right_cmd(res_l)
           record_r = res_l
           print(f"left:{res_l}")
         
        global frrame_ctr
        if frrame_ctr>=6 and frrame_ctr<7:
            left_cmd(res_r)
            print(f"right:{res_r}")
        elif frrame_ctr>7:
            frrame_ctr =0

        #print(f"left:{res_l}")
        #print(f"right:{res_r}")

    else:
        res = None


def main():

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        num_hands=2)

    with GestureRecognizer.create_from_options(options) as recognizer:

        cap = cv2.VideoCapture(0)
        start_time = time.time()
        # FPS update time in seconds
        display_time = 2
        fc = 0
        FPS = 0




        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

            while cap.isOpened():

                global frrame_ctr
                success, frame = cap.read()
                frrame_ctr+=1
                

                fc+=1
                TIME = time.time() - start_time
                if (TIME) >= display_time:
                    FPS = fc / (TIME)
                    fc = 0
                    start_time = time.time()
                
                fps_disp = "FPS: "+str(FPS)[:5]
                
                
                frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                recognizer.recognize_async(image, int(frame_timestamp_ms))

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                
                #frame = cv2.putText(frame, fps_disp, (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('rps', cv2.flip(frame, 1))
                #cv2.imshow('rps', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()


if __name__ == '__main__':
    main()
