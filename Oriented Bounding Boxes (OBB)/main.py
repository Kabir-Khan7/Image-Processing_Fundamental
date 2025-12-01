from ultralytics import YOLO

model = YOLO("yolov8s-obb.pt")

results = model.track(source="ship2.mp4", show=True, save=True, verbose=False)