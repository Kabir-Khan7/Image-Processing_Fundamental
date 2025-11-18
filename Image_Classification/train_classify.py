# train_classify.py  ← FINAL WORKING VERSION
from ultralytics import YOLO

# Load YOLOv8 classification model
model = YOLO("yolov8n-cls.pt")  # or yolov8s-cls.pt for better accuracy

print("Model downloaded and loaded!")

# Train directly on the folder containing class subfolders
model.train(
    data=".",               # "." means current folder (where alzheimer/normal/tumor are)
    epochs=50,
    imgsz=224,
    batch=8,
    patience=10,
    device="",              # auto CPU/GPU
    project="runs",         # saves here
    name="mri_classifier", # final path: runs/mri_classifier/weights/best.pt
    exist_ok=True,
    pretrained=True,
    verbose=True,
    plots=True
)

print("Training DONE!")
print("Your trained model is at: runs/mri_classifier/weights/best.pt")