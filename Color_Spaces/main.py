import os
import cv2

folder = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(folder, 'bird_py.jpg')

img = cv2.imread(image_path)
print(img.shape)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )

cv2.imshow('img_gray', img_gray)
cv2.imshow('img_hsv', img_hsv)
cv2.imshow('img', img)
cv2.imshow('img_rgb', img_rgb)
cv2.waitKey(0)