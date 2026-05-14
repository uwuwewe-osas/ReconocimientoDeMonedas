import cv2
import numpy as np
import math

def test_morph(img_path):
    img = cv2.imread(img_path)
    if img is None: return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Try different kernels
    kernels = [5, 9, 15]
    for k in kernels:
        kernel = np.ones((k,k), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if 50000 < cv2.contourArea(c) < 300000]
        
        print(f"--- {img_path} | Kernel {k} ---")
        areas = sorted([cv2.contourArea(c) for c in valid])
        for a in areas:
            print(f"ContourArea: {a:.1f}")

test_morph('Pruebas/monedasVerde.jpg')
test_morph('MonedasHallar/monedas (1).jpg')
