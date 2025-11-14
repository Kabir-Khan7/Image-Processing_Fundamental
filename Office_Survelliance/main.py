import cv2
import os
import csv
import time
from ultralytics import YOLO

# 1. CONFIG
VIDEO_IN = "office_video2.mp4"
VIDEO_OUT = "output_surveillance.mp4"
CONF_THRESH = 0.5
MODEL_NAME = "yolov8n.pt"

IDLE_THRESHOLD = 10          # Seconds with no people before idle alert
CROWD_THRESHOLD = 5          # Crowding alert if count exceeds this number

# 2. LOAD YOLO MODEL
model = YOLO(MODEL_NAME)
print(f"YOLOv8 loaded: {MODEL_NAME}")

# 3. VIDEO SETUP
if not os.path.exists(VIDEO_IN):
    raise FileNotFoundError(f"Video not found: {VIDEO_IN}")

cap = cv2.VideoCapture(VIDEO_IN)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

print(f"Processing {VIDEO_IN} → {VIDEO_OUT}")
print(f"Resolution: {w}x{h} | FPS: {fps:.1f}")

# 4. INITIALIZE VARIABLES
frame_idx = 0
total_persons = 0

idle_start = None
last_seen_time = time.time()
alert_triggered = False

csv_data = []

# 5. DETECTION LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    timestamp = frame_idx / fps

    # Run YOLO (detect persons only)
    results = model(frame, conf=CONF_THRESH, classes=[0], verbose=False)[0]
    boxes = results.boxes
    current_count = len(boxes) if boxes is not None else 0
    total_persons += current_count

    #IMPROVED ALERT LOGIC
    now = time.time()
    alert_message = ""
    alert_color = (0, 0, 255)  # Red default

    # 1) Idle Alert – No activity for long
    if current_count == 0:
        if idle_start is None:
            idle_start = now
        else:
            idle_duration = now - idle_start
            if idle_duration >= IDLE_THRESHOLD:
                alert_message = f"⚠️  NO ACTIVITY for {IDLE_THRESHOLD}s"
                alert_color = (0, 255, 255)  # Yellow
    else:
        idle_start = None
        last_seen_time = now

    # 2) Crowd Alert – Too many people
    if current_count >= CROWD_THRESHOLD:
        alert_message = f"🚨 CROWD ALERT! ({current_count} People)"
        alert_color = (0, 0, 255)  # Red alert
        alert_triggered = True

    # 3) High Activity Warning (just below crowd level)
    elif current_count >= (CROWD_THRESHOLD - 2) and current_count > 0:
        alert_message = f"⚠️ High Activity ({current_count})"
        alert_color = (0, 165, 255)  # Orange

    # DRAWING BOUNDING BOXES
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])

            # Styled bounding box (Green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)

            # Label
            cv2.putText(frame, f"Person {conf:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # ON-SCREEN TEXT OVERLAY (Clean & Readable)
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Persons: {current_count}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if alert_message:
        cv2.putText(frame, alert_message, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, alert_color, 3)

    # -------------------------------------------------
    # SAVE & DISPLAY
    # -------------------------------------------------
    out.write(frame)
    disp = cv2.resize(frame, (960, 540))
    cv2.imshow("Office Surveillance – Press Q", disp)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

    # Save raw for CSV
    csv_data.append({
        "frame": frame_idx,
        "timestamp_sec": round(timestamp, 2),
        "person_count": current_count,
        "alert": alert_message
    })

# 6. CLEANUP & SUMMARY
cap.release()
out.release()
cv2.destroyAllWindows()

avg_per_frame = total_persons / frame_idx if frame_idx else 0

print("\n" + "=" * 60)
print("SURVEILLANCE COMPLETE!")
print(f"Output: {VIDEO_OUT}")
print(f"Average persons/frame: {avg_per_frame:.2f}")

if alert_triggered:
    print("⚠️ Alerts were triggered during surveillance.")
else:
    print("✅ No alerts detected.")

print("=" * 60)

# SAVE CSV
csv_out = "surveillance_log.csv"
with open(csv_out, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["frame", "timestamp_sec", "person_count", "alert"])
    writer.writeheader()
    writer.writerows(csv_data)

print(f"Detection log saved as: {csv_out}")
print("Done.")