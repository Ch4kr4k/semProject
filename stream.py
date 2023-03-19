import mediapipe as mp
import cv2


model_path = '/media/zera/DATA/Project/semProject/rock_paper_siccor.task'


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    top_gesture=result.gestures
    

    for  element in (top_gesture):
        print(element[0].category_name)
   
    #print('gesture recognition result: {}'.format(top_gesture))

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with GestureRecognizer.create_from_options(options) as recognizer:
    
    cap = cv2.VideoCapture(0)
    

    while cap.isOpened():
       
       
       success, frame= cap.read()

       frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)      
       
       
       

       image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

       
       
       
       
       

       recognizer.recognize_async(image,int(frame_timestamp_ms))
       
       

       #top_gesture = recognition_result.gestures[0][0]

       #print(top_gesture.category_name)


       cv2.imshow('MediaPipe Face Mesh', cv2.flip(frame, 1))
       if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()
