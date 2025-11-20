from ultralytics import YOLO

#Load a model
model = YOLO("yolov8n-pose.pt")

#Predict with the model
results= model.track(source="people.mp4", show=True, save=True, verbose=True)
