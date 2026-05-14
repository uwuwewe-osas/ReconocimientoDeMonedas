import cv2
import glob
import os
import numpy as np

def clasificar_moneda(area):
    if area < 102000:
        return "10 centimos", 0.1
    elif area < 111500:
        return "2 soles", 2.0
    elif area < 118500:
        return "50 centimos", 0.5
    elif area < 136000:
        return "20 centimos", 0.2
    elif area < 156000:
        return "5 soles", 5.0
    else:
        return "1 sol", 1.0

def load_real_values():
    archivo_reales = 'valores_reales.txt'
    valores_reales = {}
    if os.path.exists(archivo_reales):
        with open(archivo_reales, mode='r', encoding='utf-8') as f:
            lineas = f.readlines()
            imagen_actual = None
            for linea in lineas:
                linea = linea.strip()
                if not linea: continue
                if linea.endswith('.jpg'):
                    imagen_actual = linea
                elif linea.startswith('TOTAL REAL:') and imagen_actual:
                    partes = linea.split(':')
                    if len(partes) > 1:
                        valor_str = partes[1].replace('soles', '').strip()
                        try: valores_reales[imagen_actual] = float(valor_str)
                        except: pass
                    imagen_actual = None
    return valores_reales

valores_reales = load_real_values()
imagenes = glob.glob('MonedasHallar/*.jpg')
error_total = 0

for img_path in imagenes:
    img = cv2.imread(img_path)
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    # Cambiamos block size a 51
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_monto = 0.0
    for c in contours:
        area_contorno = cv2.contourArea(c)
        if 50000 < area_contorno < 300000:
            _, valor = clasificar_moneda(area_contorno)
            total_monto += valor
            
    total_monto = round(total_monto, 2)
    nombre_img = os.path.basename(img_path)
    real = valores_reales.get(nombre_img, 0.0)
    err = abs(real - total_monto)
    if err > 0.01:
        error_total += 1

print(f"Total imágenes con error (BS=51): {error_total} de {len(imagenes)}")
