 🧠 Image Processing Fundamentals using OpenCV

**Author:** *Kabir Khurshid*  
🎓 *Aspiring Computer Vision Engineer | ADP in Accounting & Finance*  
📘 Repository demonstrating the **core mathematical and programming foundations of image processing** using **OpenCV (cv2)** and **Python**.

---

## 📖 Overview

This repository provides a **conceptual and practical walkthrough** of key image processing operations — from color spaces to thresholding.  
Each module includes:
- 🧩 **Mathematical foundation**
- 💻 **Python code explanation**
- 🖼️ **Visual example (input/output)**
- ⚙️ **OpenCV function references**

---

## 🧰 Requirements

Install dependencies before running:

```bash
pip install opencv-python numpy matplotlib

📂 Repository Contents
#	Topic	Description
1️⃣	Color Spaces
How images are represented and transformed between color models
2️⃣	Contours
Detecting and analyzing object boundaries
3️⃣	Edge Detection
Gradient-based methods to identify sharp intensity changes
4️⃣	Image Drawing
Annotating and drawing geometric shapes on images
5️⃣	Image Resizing
Changing image resolution and aspect ratio
6️⃣	Image Blurring
Reducing noise and detail using filters
7️⃣	Thresholding
Segmenting images into binary form
1️⃣ Color Spaces
🧠 Concept

A color space defines how colors are represented.
An image is a 3D matrix:
I(x,y)=[B(x,y),G(x,y),R(x,y)]
I(x,y)=[B(x,y),G(x,y),R(x,y)]

where each pixel has three intensity values between 0–255.

Common spaces:

    BGR / RGB – standard for color display

    GRAY – intensity only, I=0.299R+0.587G+0.114BI=0.299R+0.587G+0.114B

    HSV – Hue, Saturation, Value; used in color-based filtering

💻 Code

img = cv2.imread('bird_py.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

🖼️ Example
BGR	GRAY	HSV
Original bird image	Intensity-only	Hue-based color representation
2️⃣ Contours
🧠 Concept

Contours represent boundaries of objects with the same intensity.
They are detected by binary thresholding + contour tracing.

Mathematically:
C={(x,y)∣I(x,y)=T}
C={(x,y)∣I(x,y)=T}
💻 Code

ret, thresh = cv2.threshold(img_gray, 58, 220, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    if cv2.contourArea(cnt) > 40:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

🖼️ Example
Original	Binary	Contours
Birds in sky	Thresholded	Objects bounded by green rectangles
3️⃣ Edge Detection
🧠 Concept

Edges are locations of sharp intensity change — mathematically captured via image gradients:
Gx=∂I∂x,Gy=∂I∂y,and∣G∣=Gx2+Gy2
Gx​=∂x∂I​,Gy​=∂y∂I​,and∣G∣=Gx2​+Gy2​
​

Canny Edge Detection uses:

    Gaussian smoothing

    Gradient calculation

    Non-maximum suppression

    Double threshold & edge tracking

💻 Code

img_edge = cv2.Canny(img, 200, 550)
img_dilate = cv2.dilate(img_edge, np.ones((3,3), np.uint8))
img_erode = cv2.erode(img_dilate, np.ones((3,3), np.uint8))

🖼️ Example
Original	Edges	Dilated	Eroded
4️⃣ Image Drawing
🧠 Concept

You can annotate images using geometric primitives such as lines, rectangles, circles, and text.
Each drawing function modifies pixel values at given coordinates.
💻 Code

cv2.line(img, (100,150), (300,450), (0,255,0), 3)
cv2.rectangle(img, (200,350), (450,600), (0,0,255), 5)
cv2.circle(img, (400,200), 50, (255,0,0), 10)
cv2.putText(img, 'Hello, World!', (100,300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 5)

🖼️ Example
Operation	Result
Draw shapes and text	Annotated whiteboard
5️⃣ Image Resizing
🧠 Concept

Resizing changes image dimensions using interpolation methods:

    Nearest-neighbor (fast, low quality)

    Bilinear / Bicubic (smooth transitions)

    Area / Lanczos (for downscaling)

💻 Code

resized_img = cv2.resize(img, (450, 270))

🖼️ Example
Original	Resized
1080×720	450×270
6️⃣ Image Blurring
🧠 Concept

Blurring reduces noise and detail using filters that average pixel neighborhoods.
I′(x,y)=1k2∑i=−k/2k/2∑j=−k/2k/2I(x+i,y+j)
I′(x,y)=k21​i=−k/2∑k/2​j=−k/2∑k/2​I(x+i,y+j)
💻 Code

k_size = 25
img_blur = cv2.blur(img, (k_size, k_size))
img_gauss = cv2.GaussianBlur(img, (k_size, k_size), 5)
img_median = cv2.medianBlur(img, k_size)

🖼️ Example
Original	Average	Gaussian	Median
7️⃣ Thresholding
🧠 Concept

Thresholding converts grayscale images into binary form by comparing each pixel intensity I(x,y)I(x,y) to a threshold TT:
I′(x,y)={255,if I(x,y)>T0,otherwise
I′(x,y)={255,0,​if I(x,y)>Totherwise​

Types:

    Simple Threshold

    Adaptive Threshold

    Otsu’s Method

💻 Code

# Simple
ret, thresh = cv2.threshold(image_gray, 80, 255, cv2.THRESH_BINARY)

# Adaptive
thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 21, 10)

🖼️ Example
Original	Grayscale	Binary	Adaptive
🧩 Conceptual Flow

Image → Color Conversion → Filtering → Thresholding → Contours/Edges → Visualization

This logical sequence forms the foundation of Computer Vision pipelines used in:

    Object Detection

    Image Segmentation

    Feature Extraction

💡 Future Enhancements

    Add histogram equalization and morphological transformations

    Create interactive Jupyter notebooks

    Introduce real-time camera input examples

🙌 Author

Kabir Khurshid
📘 ADP in Accounting & Finance | Aspiring Computer Vision Engineer
🌐 GitHub Profile

“Every pixel tells a story — learn to read it.”