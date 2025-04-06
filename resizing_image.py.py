#Resizing the image 

import os
import cv2

# This gets the folder where your Python file is
folder = os.path.dirname(os.path.abspath(__file__))

# This creates the path to your image
image_path = os.path.join(folder, 'image_py.jpg')

# Load the image
img = cv2.imread(image_path)
resized_img = cv2.resize(img, (450, 270))

#print the image shape
print(img.shape)
print(resized_img.shape)

#show the image
cv2.imshow('img', img)
cv2.imshow('resized_img', resized_img)
cv2.waitKey(0)