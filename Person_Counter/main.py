import cv2
import os
from ultralytics import YOLO

# ------------------------------
# CONFIGURATION
# ------------------------------
VIDEO_PATH = "people2.mp4"        # Your video file
OUTPUT_FOLDER = "processed_frames"    # Folder to save processed frames
FRAME_FORMAT = "jpg"                  # jpg / png
MODEL_PATH = "yolov8n.pt"             # YOLO model file
# ------------------------------

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLO model
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)

frame_count = 0
saved_frame_id = 0

print("[INFO] Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Capture only 1 frame per second
    if frame_count % frame_interval == 0:
        saved_frame_id += 1

        # Run YOLO detection
        results = model(frame, verbose=False)

        person_count = 0

        # Draw YOLO detections
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label == "person":
                    person_count += 1

                    # bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)

        # --------------------------
        # NEW FEATURE: Show total count on frame
        # --------------------------
        cv2.putText(frame,
                    f"Total Persons: {person_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3)
        # --------------------------

        # Save frame
        filename = f"frame_{saved_frame_id}.{FRAME_FORMAT}"
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(save_path, frame)

        print(f"[FRAME {saved_frame_id}] Persons Detected: {person_count} → Saved: {filename}")

    frame_count += 1

cap.release()
print("[INFO] Processing completed.")
