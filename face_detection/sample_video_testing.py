import cv2

video_path = 'sample_video.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

# Get the Frames Per Second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the delay in milliseconds (1000ms / FPS)
frame_delay = int(1000 / fps) 
# Note: For many videos, this will be around 33ms (for 30 FPS) or 40ms (for 25 FPS)
print(f"Video FPS detected: {fps:.2f}. Setting delay to {frame_delay}ms.")
# ----------------------------------------------------

while True:
    ret, frame = cap.read()

    if ret:
        cv2.imshow('Video Feed (Press Q to exit)', frame)

        # Apply the calculated delay here
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break
    else:
        print("Video playback finished.")
        break

cap.release()
cv2.destroyAllWindows()