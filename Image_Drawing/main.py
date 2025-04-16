import os 
import cv2

folder = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(folder, 'whiteboard.jpg')

img = cv2.imread(image_path)
print(img.shape)

#line
cv2.line(img, (100, 150), (300, 450), (0, 255, 0), 3)
#rectanlge
cv2.rectangle(img, (200, 350), (450, 600), (0, 0, 255), 5)
#circle
cv2.circle(img, (400, 200), 50, (255, 0, 0), 10)
#Text
cv2.putText(img, 'Hello, World!', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)

cv2.imshow('img', img)
cv2.waitKey(0)
