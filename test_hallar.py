import cv2
import numpy as np

img_path = 'MonedasHallar/monedas (1).jpg'
img = cv2.imread(img_path)

# Print info
print("Image shape:", img.shape)
print("Corner color:", img[0,0])

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (15, 15), 0)

# Adaptive threshold
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid_contours = [c for c in contours if 1000 < cv2.contourArea(c) < 3000000]

print(f"Detected {len(valid_contours)} coins using adaptive threshold")
for i, c in enumerate(valid_contours[:10]):  # print max 10
    area = cv2.contourArea(c)
    (x,y), radius = cv2.minEnclosingCircle(c)
    print(f"Coin {i+1}: Area = {area:.2f}, Diameter = {radius*2:.2f}")
