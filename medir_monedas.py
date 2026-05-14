import cv2
import os
import glob
import numpy as np
import math

path = 'MonedasUnidad/*.jpg'
for img_path in glob.glob(path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if 1000 < cv2.contourArea(c) < 3000000]
    
    if len(valid_contours) > 0:
        c = max(valid_contours, key=cv2.contourArea)
        area_contorno = cv2.contourArea(c)
        (x,y), radius = cv2.minEnclosingCircle(c)
        area_circulo = math.pi * (radius**2)
        print(f"{os.path.basename(img_path)}: Contorno = {area_contorno:.2f}, Circulo = {area_circulo:.2f}")
