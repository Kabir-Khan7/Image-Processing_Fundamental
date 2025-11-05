import os
import cv2

# This gets the folder where your Python file is
folder = os.path.dirname(os.path.abspath(__file__))

# This creates the path to your image
image_path = os.path.join(folder, 'image_py.jpg')

# Load the image
img = cv2.imread(image_path)
print(img.shape)

#Cropped image
cropped_img = img[200:400, 350:650]

cv2.imshow('img', img)
cv2.imshow('cropped_img', cropped_img)
cv2.waitKey(0)