
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

landmark_list =[
                        mp_hands.HandLandmark(0).name,
                        mp_hands.HandLandmark(1).name,
                        mp_hands.HandLandmark(2).name,
                        mp_hands.HandLandmark(3).name,
                        mp_hands.HandLandmark(4).name,
                        mp_hands.HandLandmark(5).name,
                        mp_hands.HandLandmark(6).name,
                        mp_hands.HandLandmark(7).name,
                        mp_hands.HandLandmark(8).name,
                        mp_hands.HandLandmark(9).name,
                       mp_hands.HandLandmark(10).name,
                        mp_hands.HandLandmark(11).name,
                        mp_hands.HandLandmark(12).name,
                        mp_hands.HandLandmark(13).name,
                        mp_hands.HandLandmark(14).name,
                        mp_hands.HandLandmark(15).name,
                        mp_hands.HandLandmark(16).name,
                        mp_hands.HandLandmark(17).name,
                        mp_hands.HandLandmark(18).name,
                        mp_hands.HandLandmark(19).name,
                        mp_hands.HandLandmark(20).name                       

                ]

	
	
print(landmark_list)	
cap = cv2.VideoCapture(0)  # 2 or 1 for webcam
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference dont really know shit what this writable do.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)




  
    
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

        print(f'{hand_landmarks.landmark[mp_hands.HandLandmark(5).value]}')
        #print(f'{hand_landmarks.landmark[mp_hands.HandLandmark(5).value].x}')
        #print(f'{mp_hands.HandLandmark(5).name}')
        
                
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    keyCode = cv2.waitKey(100)
    if (keyCode & 0xFF) == ord("q"):
      break
cap.release()
