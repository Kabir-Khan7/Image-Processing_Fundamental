import cv2

# Load image
img = cv2.imread('people_image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,      # Image scale per level
    minNeighbors=5,       # How many neighbors each candidate must have
    minSize=(30, 30)      # Minimum face size
)

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

# Show result
cv2.imshow('Faces Detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()