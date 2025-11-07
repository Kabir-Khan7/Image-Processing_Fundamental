# yolo_car_detection.py – Accurate car detection using YOLOv8
import cv2
import os
from ultralytics import YOLO

# -------------------------------------------------
# 1. CONFIG
# -------------------------------------------------
VIDEO_IN   = "car1.mp4"
VIDEO_OUT  = "output_yolo_cars.mp4"
CONF_THRESH = 0.5   # Confidence threshold (0.3–0.7)
MODEL_NAME = "yolov8n.pt"  # Nano (fast) or yolov8s.pt (more accurate)

# -------------------------------------------------
# 2. Load YOLOv8 (auto-downloads if missing)
# -------------------------------------------------
model = YOLO(MODEL_NAME)  # Downloads ~6MB on first run
print(f"YOLOv8 loaded: {MODEL_NAME}")

# -------------------------------------------------
# 3. Video I/O
# -------------------------------------------------
if not os.path.exists(VIDEO_IN):
    raise FileNotFoundError(f"Video not found: {VIDEO_IN}")

cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise IOError("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

print(f"Processing {VIDEO_IN} → {VIDEO_OUT}")
print(f"Resolution: {w}x{h} | FPS: {fps:.1f}")

# -------------------------------------------------
# 4. Car class IDs in COCO (YOLOv8)
# -------------------------------------------------
CAR_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# -------------------------------------------------
# 5. Detection Loop
# -------------------------------------------------
frame_idx = 0
total_vehicles = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    progress = (frame_idx / total) * 100 if total else 0

    # YOLO inference
    results = model(frame, conf=CONF_THRESH, classes=list(CAR_CLASSES.keys()), verbose=False)[0]
    boxes = results.boxes

    cur_vehicles = 0
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = box.conf[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            label = CAR_CLASSES.get(cls_id, "vehicle")

            cur_vehicles += 1
            total_vehicles += 1

            # Draw
            color = (0, 255, 0) if label == "car" else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.0%}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Stats
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Vehicles: {cur_vehicles}", (10, 60),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Progress: {progress:.1f}%", (10, 90),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 2)

    # Save & show
    out.write(frame)
    disp = cv2.resize(frame, (960, 540))
    cv2.imshow("YOLOv8 Car Detection – Press Q", disp)

    if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
        break

# -------------------------------------------------
# 6. Cleanup
# -------------------------------------------------
cap.release()
out.release()
cv2.destroyAllWindows()

avg = total_vehicles / frame_idx if frame_idx else 0
print(f"\n{'='*60}")
print("YOLOv8 DETECTION COMPLETE!")
print(f"Output: {VIDEO_OUT}")
print(f"Total vehicles: {total_vehicles}")
print(f"Avg per frame: {avg:.2f}")
print(f"{'='*60}")