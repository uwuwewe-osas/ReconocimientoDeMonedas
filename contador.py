import cv2
import os
import glob
import numpy as np
import csv

def clasificar_moneda(area):
    """
    Clasifica la moneda basado en el área del contorno.
    Los valores de umbral (X, Y, etc.) fueron justificados empíricamente
    midiendo las imágenes de referencia en la carpeta 'MonedasUnidad'.
    
    Áreas de referencia obtenidas:
    - 10 céntimos: ~95,194
    - 2 soles: ~109,131
    - 50 céntimos: ~114,722
    - 20 céntimos: ~123,204
    - 5 soles: ~149,628
    - 1 sol: ~163,456
    """
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

def main():
    carpeta_hallar = 'MonedasHallar'
    carpeta_salida = 'MonedasContadas'
    os.makedirs(carpeta_salida, exist_ok=True)
    imagenes = glob.glob(os.path.join(carpeta_hallar, '*.jpg'))
    
    # Cargar valores reales si existen en formato TXT
    archivo_reales = 'valores_reales.txt'
    valores_reales = {}
    if os.path.exists(archivo_reales):
        with open(archivo_reales, mode='r', encoding='utf-8') as f:
            lineas = f.readlines()
            
            imagen_actual = None
            for linea in lineas:
                linea = linea.strip()
                if not linea:
                    continue
                if linea.endswith('.jpg'):
                    imagen_actual = linea
                elif linea.startswith('TOTAL REAL:') and imagen_actual:
                    partes = linea.split(':')
                    if len(partes) > 1:
                        # Extraer solo el número
                        valor_str = partes[1].replace('soles', '').strip()
                        try:
                            valores_reales[imagen_actual] = float(valor_str)
                        except ValueError:
                            pass
                    imagen_actual = None
    
    print(f"{'Imagen':<20} | {'Real':<6} | {'Detectado':<9} | {'Error':<6}")
    print("-" * 50)
    
    for img_path in imagenes:
        nombre_img = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_mostrar = img.copy()
        
        # 1. Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Aplicar blur
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # 3. Binarizar usando Adaptive Thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Operaciones morfológicas para rellenar huecos
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # 4. Detectar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_monto = 0.0
        
        for c in contours:
            area = cv2.contourArea(c)
            # Filtrar contornos que no son monedas
            if 50000 < area < 300000:
                # Centroides marcados
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                
                # Clasificar moneda
                tipo, valor = clasificar_moneda(area)
                total_monto += valor
                
                # Dibujar contorno y centroide
                cv2.drawContours(img_mostrar, [c], -1, (0, 255, 0), 3)
                cv2.circle(img_mostrar, (cX, cY), 7, (0, 0, 255), -1)
                
                # Poner texto (tipo de moneda)
                cv2.putText(img_mostrar, tipo, (cX - 40, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        total_monto = round(total_monto, 2)
        
        # Total en pantalla
        cv2.putText(img_mostrar, f"TOTAL: {total_monto} soles", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        
        # Evaluar error
        real = valores_reales.get(nombre_img, 0.0)
        error = abs(real - total_monto)
        
        print(f"{nombre_img:<20} | {real:<6.2f} | {total_monto:<9.2f} | {error:<6.2f}")
        
        # Guardar imagen en la nueva carpeta
        ruta_salida = os.path.join(carpeta_salida, nombre_img)
        cv2.imwrite(ruta_salida, img_mostrar)

if __name__ == '__main__':
    main()
