# surveillance_person_monitoring.py – Smart Office Surveillance using YOLOv8
import cv2
import os
import csv
import time
from ultralytics import YOLO

# -------------------------------------------------
# 1. CONFIG
# -------------------------------------------------
VIDEO_IN = "office_video.mp4"   # Replace with your office video
VIDEO_OUT = "output_surveillance.mp4"
CONF_THRESH = 0.5
MODEL_NAME = "yolov8n.pt"
IDLE_THRESHOLD = 10      # Seconds with no people before alert
CROWD_THRESHOLD = 5      # Number of people to trigger "crowd alert"

# -------------------------------------------------
# 2. LOAD MODEL
# -------------------------------------------------
model = YOLO(MODEL_NAME)
print(f"YOLOv8 loaded: {MODEL_NAME}")

# -------------------------------------------------
# 3. VIDEO SETUP
# -------------------------------------------------
if not os.path.exists(VIDEO_IN):
    raise FileNotFoundError(f"Video not found: {VIDEO_IN}")

cap = cv2.VideoCapture(VIDEO_IN)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))
print(f"Processing {VIDEO_IN} → {VIDEO_OUT}")
print(f"Resolution: {w}x{h} | FPS: {fps:.1f}")

# -------------------------------------------------
# 4. DETECTION VARIABLES
# -------------------------------------------------
frame_idx = 0
total_persons = 0
last_seen_time = time.time()
idle_start = None
alert_triggered = False
csv_data = []

# -------------------------------------------------
# 5. DETECTION LOOP
# -------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    timestamp = frame_idx / fps  # in seconds

    # Run YOLO detection
    results = model(frame, conf=CONF_THRESH, classes=[0], verbose=False)[0]
    boxes = results.boxes
    current_count = len(boxes) if boxes is not None else 0
    total_persons += current_count

    # Alert logic
    now = time.time()
    alert_message = ""
    if current_count > 0:
        last_seen_time = now
        idle_start = None
    else:
        if idle_start is None:
            idle_start = now
        elif now - idle_start > IDLE_THRESHOLD:
            alert_message = f"⚠️ Idle Alert! No people for {IDLE_THRESHOLD}s"
            alert_triggered = True

    if current_count > CROWD_THRESHOLD:
        alert_message = f"🚨 Crowd Alert! {current_count} persons detected"
        alert_triggered = True

    # Draw bounding boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Person {conf:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display info
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Persons: {current_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if alert_message:
        cv2.putText(frame, alert_message, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Save and show
    out.write(frame)
    disp = cv2.resize(frame, (960, 540))
    cv2.imshow("Office Surveillance – Press Q", disp)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

    # Save log data
    csv_data.append({
        "frame": frame_idx,
        "timestamp_sec": round(timestamp, 2),
        "person_count": current_count,
        "alert": alert_message
    })

# -------------------------------------------------
# 6. CLEANUP & REPORT
# -------------------------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

avg_per_frame = total_persons / frame_idx if frame_idx else 0
print(f"\n{'=' * 60}")
print("SURVEILLANCE COMPLETE!")
print(f"Output: {VIDEO_OUT}")
print(f"Average persons/frame: {avg_per_frame:.2f}")
if alert_triggered:
    print("⚠️ Alerts were triggered during surveillance.")
else:
    print("✅ No alerts detected.")
print(f"{'=' * 60}")

# Save CSV log
csv_out = "surveillance_log.csv"
with open(csv_out, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["frame", "timestamp_sec", "person_count", "alert"])
    writer.writeheader()
    writer.writerows(csv_data)
print(f"Detection log saved as: {csv_out}")
