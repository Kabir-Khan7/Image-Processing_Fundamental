import os 
import cv2
import numpy as np

folder = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(folder, 'basket_ball_player.jpg')

img = cv2.imread(image_path)
print(img.shape)
img_edge = cv2.Canny(img, 200, 550)

#to make thick the canny edge image 
img_dilate = cv2.dilate(img_edge, np.ones((3, 3), dtype=np.int8))

#this erode function do the opposite 
imge_erode = cv2.erode(img_dilate, np.ones((3, 3), dtype=np.int8))

cv2.imshow('img', img)
cv2.imshow('img_edge', img_edge)
cv2.imshow('img_dilate', img_dilate)
cv2.imshow('imge_erode', imge_erode)

cv2.waitKey(0)

#1:24 (Open cv tuturial)