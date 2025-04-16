#Compile result of both adaptive threshold and global threshold 

import os 
import cv2

folder = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(folder, 'bear.jpg')

img = cv2.imread(image_path)

image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

adaptive_thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)
ret, simple_thresh = cv2.threshold(image_gray, 80, 255, cv2.THRESH_BINARY)

cv2.imshow('img', img)
cv2.imshow('adaptive_thresh', adaptive_thresh)          
cv2.imshow('simple_thresh', simple_thresh)          
cv2.waitKey(0)