#Thresholding is a technique used to segment an image by turning it into a binary image
import os 
import cv2

folder = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(folder, 'bear.jpg')

img = cv2.imread(image_path)

image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(image_gray, 80, 255, cv2.THRESH_BINARY)
thresh = cv2.blur(thresh, (10, 10))
ret, thresh = cv2.threshold(thresh, 80, 255, cv2.THRESH_BINARY)

cv2.imshow('img', img)
cv2.imshow('image_gray', image_gray)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)