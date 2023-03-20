import mediapipe as mp
import cv2
import pyautogui as key
from ges_cmd import ges_to_cmd

model_path = './model.task'

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

    if gestures1 == 'rock':
        key.press(ges_to_cmd[gestures1])


def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    
    top_gesture = result.gestures
    res = ''

    if len(top_gesture) == 1:
        res = top_gesture[0][0].category_name
        res_l = ''
        res_r = ''
        print(res)

    elif len(top_gesture) == 2:
        res_r = top_gesture[0][0].category_name  # right -> left in reality
        res_l = top_gesture[1][0].category_name  # left -> right in reality
        print(f"left:{res_l}")
        print(f"right:{res_r}")
        
    else:
        res = ''

    key_cmd(res)


def main():
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        num_hands=2)

    with GestureRecognizer.create_from_options(options) as recognizer:

        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

            while cap.isOpened():
                success, frame = cap.read()
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

                cv2.imshow('rps', cv2.flip(frame, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()

if __name__ == '__main__':
    main()