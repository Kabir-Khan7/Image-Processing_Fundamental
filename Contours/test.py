import os 
import cv2

folder = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(folder, 'birds_in_sky.jpg')

img = cv2.imread(image_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(img_gray, 58, 220, cv2.THRESH_BINARY_INV)
contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 40:
        # Uncomment this line if needed
        # cv2.drawContours(img, cnt, -1, (0, 255, 0), 1)
        
        x1, y1, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.imshow('img_gray', img_gray)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
