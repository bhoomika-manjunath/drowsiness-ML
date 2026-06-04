import cv2
import time
import math
import serial
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------------------
# SETTINGS
# ----------------------------
SERIAL_PORT = "COM3"          # change only if your ESP32 port changes
BAUD_RATE = 9600
CAMERA_INDEX = 1              # use 1 because that worked for you
MODEL_PATH = "face_landmarker.task"
DROWSY_LIMIT = 2.0            # seconds
EAR_THRESHOLD = 0.21          # tune if needed

# Eye landmark indices used for EAR calculation
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def eye_aspect_ratio(pts):
    a = dist(pts[1], pts[5])
    b = dist(pts[2], pts[4])
    c = dist(pts[0], pts[3])
    if c == 0:
        return 0.0
    return (a + b) / (2.0 * c)

# ----------------------------
# SERIAL SETUP
# ----------------------------
try:
    esp32 = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    esp32.reset_input_buffer()
    print("ESP32 connected")
except Exception as e:
    esp32 = None
    print("ESP32 not connected:", e)

# ----------------------------
# CAMERA SETUP
# ----------------------------
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera not opened")
    raise SystemExit

# ----------------------------
# MEDIAPIPE SETUP
# ----------------------------
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
RunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

closed_start_time = None
alert_sent = False

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        eyes_open = False
        ear_value = 0.0
        closed_seconds = 0.0

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]

            # Convert normalized landmarks to pixel coordinates
            px = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]

            left_eye_pts = [px[i] for i in LEFT_EYE]
            right_eye_pts = [px[i] for i in RIGHT_EYE]

            left_ear = eye_aspect_ratio(left_eye_pts)
            right_ear = eye_aspect_ratio(right_eye_pts)
            ear_value = (left_ear + right_ear) / 2.0

            eyes_open = ear_value > EAR_THRESHOLD

            # Draw landmarks
            for idx in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, px[idx], 2, (255, 0, 0), -1)

        # ----------------------------
        # DROWSINESS LOGIC
        # ----------------------------
        if eyes_open:
            status = "Eyes Open"
            status_color = (0, 255, 0)
            closed_start_time = None
            alert_sent = False

            if esp32:
                esp32.write(b"0\n")

        else:
            if result.face_landmarks:
                status = "Eyes Closed"
                status_color = (0, 0, 255)

                if closed_start_time is None:
                    closed_start_time = time.time()

                closed_seconds = time.time() - closed_start_time

                if closed_seconds >= DROWSY_LIMIT:
                    if not alert_sent and esp32:
                        esp32.write(b"1\n")
                        alert_sent = True

                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )
                else:
                    if esp32:
                        esp32.write(b"0\n")
            else:
                status = "No Face Detected"
                status_color = (0, 0, 255)
                closed_start_time = None
                alert_sent = False
                if esp32:
                    esp32.write(b"0\n")

        # ----------------------------
        # DISPLAY TEXT
        # ----------------------------
        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            3
        )

        cv2.putText(
            frame,
            f"EAR: {ear_value:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Closed for: {closed_seconds:.1f} sec",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

if esp32:
    esp32.close()