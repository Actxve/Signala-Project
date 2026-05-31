import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os
import time

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH = 'hand_landmarker.task'
LANDMARK_DATA_CSV = 'landmark_data.csv'
SAMPLES_PER_SIGN = 100
SIGNS_TO_COLLECT = ['A', 'B', 'C', 'L', 'Y']
# ────────────────────────────────────────────────────────────────────────────

# ── CSV SETUP ────────────────────────────────────────────────────────────────
header = ['label']
for i in range(21):
    header += [f'x{i}', f'y{i}', f'z{i}']

file_exists = os.path.exists(LANDMARK_DATA_CSV)
csv_file = open(LANDMARK_DATA_CSV, 'a', newline='')
writer = csv.writer(csv_file)
if not file_exists:
    writer.writerow(header)
# ────────────────────────────────────────────────────────────────────────────

# ── MEDIAPIPE SETUP ──────────────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)
# ────────────────────────────────────────────────────────────────────────────

# ── WEBCAM + COLLECTION LOOP ─────────────────────────────────────────────────
cap = cv2.VideoCapture(0)

sign_index = 0
samples_collected = 0
collecting = False
last_sample_time = 0
SAMPLE_COOLDOWN = 0.05

print("Data Collection started.")
print(f"Signs to collect: {SIGNS_TO_COLLECT}")
print("Press SPACE to start collecting for the current sign.")
print("Press 'q' to quit and save.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    if sign_index >= len(SIGNS_TO_COLLECT):
        print("All signs collected!")
        break

    current_sign = SIGNS_TO_COLLECT[sign_index]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    hand_detected = bool(detection_result.hand_landmarks)

    if collecting and hand_detected:
        now = time.time()
        if now - last_sample_time >= SAMPLE_COOLDOWN:
            landmarks = detection_result.hand_landmarks[0]
            row = [current_sign]
            for lm in landmarks:
                row += [round(lm.x, 6), round(lm.y, 6), round(lm.z, 6)]
            writer.writerow(row)
            csv_file.flush()
            samples_collected += 1
            last_sample_time = now

            if samples_collected >= SAMPLES_PER_SIGN:
                print(f"  Collected {SAMPLES_PER_SIGN} samples for '{current_sign}'.")
                sign_index += 1
                samples_collected = 0
                collecting = False
                print(f"\nNext sign: '{SIGNS_TO_COLLECT[sign_index]}'" if sign_index < len(SIGNS_TO_COLLECT) else "Done!")
                print("Press SPACE to start collecting.\n")

    if hand_detected:
        for lm in detection_result.hand_landmarks[0]:
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    status = f"RECORDING ({samples_collected}/{SAMPLES_PER_SIGN})" if collecting else "READY - Press SPACE"
    hand_status = "Hand: YES" if hand_detected else "Hand: NO"
    cv2.putText(frame, f"Sign: {current_sign}  |  {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, hand_status, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if hand_detected else (0, 0, 255), 2)

    cv2.imshow('Signala - Data Collection', frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        if hand_detected:
            collecting = True
            print(f"  Recording '{current_sign}'...")
        else:
            print("  No hand detected — position your hand first.")

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"\nData saved to '{LANDMARK_DATA_CSV}'.")