# hand_tracking_test_new_api.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import time
import os

# Download model if not present
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand landmark model...")
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")

# Create hand landmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# Start webcam
cap = cv2.VideoCapture(0)
print("Starting ASL Hand Tracking (New API)...")
print("Press 'q' to quit")

fps_start_time = time.time()
fps_frame_count = 0
fps = 0

while cap.isOpened():
    success, frame = cap.read()

    fps_frame_count += 1
    elapsed = time.time() - fps_start_time
    if elapsed >= 1.0:
        fps = fps_frame_count / elapsed
        fps_frame_count = 0
        fps_start_time = time.time()
    
    if not success:
        continue
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect hands
    detection_result = detector.detect(mp_image)
    
    # Draw landmarks
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Draw all landmarks
            for landmark in hand_landmarks:
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            
            print(f"Hand detected! Landmarks: {len(hand_landmarks)}")
            
            #tacking landmark coordinates and logging them to the console
        for hand_landmarks in detection_result.hand_landmarks:
            for id, landmark in enumerate(hand_landmarks):

                #getting frame dimensions
                h, w, c = frame.shape

                #denormalizing landmark coordinates
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cz = landmark.z
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                #logging landmark coordinates to the console
                print(f"Landmark {id}: ({cx}, {cy})")
                cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    cv2.putText(frame, f"Signala - ASL Hand Tracking - FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Signala Hand Tracking Test', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()