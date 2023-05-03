import cv2
import mediapipe as mp
import csv

#csv lanmard saver
def save_landmark(landmarks):  

    for landmark in landmarks:
        writer.writerow([landmark.x, landmark.y, landmark.z, landmark.visibility])



with open('landmarks.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'z', 'visibility'])
# Create a mediapipe pose instance

mp_pose = mp.solutions.pose

# Create a mediapipe drawing instance
mp_draw = mp.solutions.drawing_utils

# Load the images from a directory
images = ["image1.jpg", "image2.jpg", "image3.jpg"]

# Initialize a list to store the landmarks of all images
landmarks_all_images = []

# Loop through each image
for image_path in images:
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect the pose landmarks in the image
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image_rgb)

        # Draw the pose landmarks on the image
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Append the landmarks to the list
        landmarks = results.pose_landmarks.landmark
        with open('landmarks.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['x', 'y', 'z', 'visibility'])
            for landmark in landmarks:
                 writer.writerow([landmark.x, landmark.y, landmark.z, landmark.visibility])

        #for landmark in landmarks:
             #print(landmark)
             #print(f"Landmark {landmark.landmark}: x={landmark.x}, y={landmark.y}, z={landmark.z}")
        
        #landmarks_all_images.append(results.pose_landmarks)

# Print the landmarks of all images

#print(landmarks_all_images)
