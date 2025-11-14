import cv2
from ultralytics import YOLO

#Load the pre-trained YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")  # Nano model (small and fast)

#Open the video
video_path = "video.mp4"  # Path to your input video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        results = model(frame)
        #Visualize the segmentation results on the frame
        annotated_frame = results[0].plot()
        cv2.imshow("Yolov8 Segmentation", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()



