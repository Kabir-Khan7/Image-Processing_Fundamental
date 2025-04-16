import os 
import cv2

folder = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(folder, 'freelancer.jpg')

img = cv2.imread(image_path)
print(img.shape) 

k_size = 25
img_blur = cv2.blur(img, (k_size, k_size))
img_gauss = cv2.GaussianBlur(img, (k_size, k_size), 5)
img_median = cv2.medianBlur(img, k_size)

cv2.imshow('img', img)
cv2.imshow('img_gauss', img_gauss)
cv2.imshow('img_median', img_median)
cv2.imshow('img_blur', img_blur)

cv2.waitKey(0)