# predict.py
from ultralytics import YOLO
import cv2
from pathlib import Path

# Load the trained model (correct path now!)
model = YOLO("runs/mri_classifier/weights/best.pt")

# Test all images
class_folders = ["alzheimer", "normal", "tumor"]

for folder in class_folders:
    print(f"\n=== Testing {folder.upper()} images ===")
    folder_path = Path(folder)
    for img_path in folder_path.glob("*.*"):  # jpg, png, etc.
        if not img_path.is_file():
            continue

        results = model(img_path, imgsz=224)[0]
        pred_class = results.names[results.probs.top1]
        confidence = results.probs.top1conf.item()

        # Display with OpenCV
        img = cv2.imread(str(img_path))
        color = (0, 255, 0) if pred_class == folder else (0, 0, 255)
        text = f"{pred_class.upper()} {confidence:.1%}"

        cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
        cv2.imshow("MRI Prediction", img)
        print(f"{img_path.name} → {pred_class.upper()} ({confidence:.1%})")

        cv2.waitKey(0)

cv2.destroyAllWindows()
print("\nAll 12 images tested!")