# advanced_lane_vehicle_detection_final.py
# Fully self-contained: Lane + Vehicle Detection using YOLOv8 + OpenCV
# -------------------------------------------------
import os
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------
# Configuration
# ----------------------
VIDEO_IN = "lane_video2.mp4"              # Input video file
VIDEO_OUT = "output_advanced_lanes_vehicles.mp4"  # Output video file
LANE_COLOR = (0, 255, 255)               # Yellow lanes
VEHICLE_COLOR = (0, 0, 255)              # Red boxes for vehicles

# ----------------------
# Load YOLO Model
# ----------------------
# You can use any trained YOLOv8 model (like 'yolov8n.pt' or your custom one)
model = YOLO("yolov8n.pt")

# ----------------------
# Lane Detection using OpenCV
# ----------------------
def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = frame.shape[:2]
    roi = np.array([[
        (100, height),
        (width - 100, height),
        (width // 2, int(height * 0.6))
    ]])
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi, 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    line_img = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), LANE_COLOR, 4)

    combined = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    return combined

# ----------------------
# Vehicle Detection using YOLOv8
# ----------------------
def detect_vehicles(frame):
    results = model(frame, verbose=False)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        if label in ["car", "truck", "bus", "motorbike"] and conf > 0.4:
            cv2.rectangle(frame, (x1, y1), (x2, y2), VEHICLE_COLOR, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, VEHICLE_COLOR, 1)

    return frame

# ----------------------
# Process Video
# ----------------------
def process_video():
    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {VIDEO_IN}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    print("🚀 Processing video...")

    gui_available = True
    try:
        cv2.imshow("test", np.zeros((10,10,3), np.uint8))
        cv2.destroyAllWindows()
    except cv2.error:
        gui_available = False
        print("[INFO] OpenCV GUI not available. Running in headless mode.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lane_frame = detect_lanes(frame)
        vehicle_frame = detect_vehicles(lane_frame)

        out.write(vehicle_frame)

        # Display only if GUI is available
        if gui_available:
            try:
                cv2.imshow("Lane & Vehicle Detection", vehicle_frame)
                if cv2.waitKey(max(1, int(1000 / fps))) & 0xFF == ord('q'):
                    print("[INFO] Exiting early by user request.")
                    break
            except cv2.error:
                gui_available = False
                print("[WARN] GUI display failed. Switching to headless mode.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"✅ Processing complete. Output saved to {VIDEO_OUT}")


if __name__ == "__main__":
    process_video()
