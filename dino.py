import keyboard as key
import pyautogui as ctrl
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def distance(x1, y1, x2, y2):

    # Calculating distance

    return (((x2 - x1)**2 + (y2 - y1)**2)**0.5)


cap = cv2.VideoCapture(0)  # 2 for webcam
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference dont really know shit what this writable do.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                tip_len = distance(float(hand_landmarks.landmark[mp_hands.HandLandmark(7).value].x), float(hand_landmarks.landmark[mp_hands.HandLandmark(
                    7).value].y), float(hand_landmarks.landmark[mp_hands.HandLandmark(0).value].x), float(hand_landmarks.landmark[mp_hands.HandLandmark(0).value].y))
                dip_len = distance(float(hand_landmarks.landmark[mp_hands.HandLandmark(8).value].x), float(hand_landmarks.landmark[mp_hands.HandLandmark(
                    8).value].y), float(hand_landmarks.landmark[mp_hands.HandLandmark(0).value].x), float(hand_landmarks.landmark[mp_hands.HandLandmark(0).value].y))

                if (tip_len + dip_len >= 0.5) and (tip_len + dip_len <= 0.7):
                    print('closed')
                    continue

                elif tip_len + dip_len >= 0.7:
                    print('open')

                    key.press_and_release('space')
                    continue
                    # time.sleep(0.001)

                else:
                    continue

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        keyCode = cv2.waitKey(10)
        if (keyCode & 0xFF) == ord("q"):
            break
cap.release()

'''
prfloat(f'{mp_hands.HandLandmark(7).name}')
        prfloat(f'{hand_landmarks.landmark[mp_hands.HandLandmark(7).value].x}')
        prfloat(f'{hand_landmarks.landmark[mp_hands.HandLandmark(7).value].y}')
        prfloat(f'{mp_hands.HandLandmark(8).name}')
        prfloat(f'{hand_landmarks.landmark[mp_hands.HandLandmark(8).value].x}')
        prfloat(f'{hand_landmarks.landmark[mp_hands.HandLandmark(8).value].y}')
        prfloat(f'{mp_hands.HandLandmark(5).name}')
'''
