import cv2
import numpy as np
import base64
import os
import glob
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Contador de Monedas")

# Create templates directory if it doesn't exist
os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

def clasificar_moneda(area):
    if area < 100000:
        return "10 centimos", 0.1
    elif area < 114000:
        return "2 soles", 2.0
    elif area < 121000:
        return "50 centimos", 0.5
    elif area < 135000:
        return "20 centimos", 0.2
    elif area < 154000:
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
                if not linea:
                    continue
                if linea.endswith('.jpg'):
                    imagen_actual = linea
                elif linea.startswith('TOTAL REAL:') and imagen_actual:
                    partes = linea.split(':')
                    if len(partes) > 1:
                        valor_str = partes[1].replace('soles', '').strip()
                        try:
                            valores_reales[imagen_actual] = float(valor_str)
                        except ValueError:
                            pass
                    imagen_actual = None
    return valores_reales

def process_image(img):
    img_mostrar = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_monto = 0.0
    monedas_detalles = []
    
    for c in contours:
        area_contorno = cv2.contourArea(c)
        if 50000 < area_contorno < 300000:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            # Use enclosing circle area for robust classification on any background
            import math
            (x,y), radius = cv2.minEnclosingCircle(c)
            area_circulo = math.pi * (radius**2)
            
            tipo, valor = clasificar_moneda(area_circulo)
            total_monto += valor
            monedas_detalles.append(tipo)
            
            cv2.drawContours(img_mostrar, [c], -1, (0, 255, 0), 3)
            cv2.circle(img_mostrar, (cX, cY), 7, (0, 0, 255), -1)
            cv2.putText(img_mostrar, tipo, (cX - 40, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    total_monto = round(total_monto, 2)
    cv2.putText(img_mostrar, f"TOTAL: {total_monto} soles", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
    
    # Convert image to base64 for web
    _, buffer = cv2.imencode('.jpg', img_mostrar)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Count occurrences
    conteo = {}
    for m in monedas_detalles:
        conteo[m] = conteo.get(m, 0) + 1
        
    return {
        "total": total_monto,
        "conteo": conteo,
        "image_base64": img_base64
    }

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    imagenes = glob.glob('MonedasHallar/*.jpg')
    archivos = [os.path.basename(img) for img in imagenes]
    archivos.sort(key=lambda x: int(x.replace('monedas (', '').replace(').jpg', '')))
    return templates.TemplateResponse(request, "index.html", {"imagenes": archivos})

@app.get("/api/analyze/{filename}")
async def analyze_dataset_image(filename: str):
    path = os.path.join('MonedasHallar', filename)
    if not os.path.exists(path):
        return {"error": "Image not found"}
        
    img = cv2.imread(path)
    if img is None:
        return {"error": "Could not read image"}
        
    result = process_image(img)
    
    # Compare with real value
    valores_reales = load_real_values()
    real_total = valores_reales.get(filename, 0.0)
    
    result["real_total"] = real_total
    result["error"] = round(abs(real_total - result["total"]), 2)
    result["is_correct"] = result["error"] == 0.0
    
    return result

@app.post("/api/upload")
async def analyze_upload(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image"}
        
    result = process_image(img)
    result["real_total"] = "N/A (Subida manual)"
    result["error"] = "N/A"
    result["is_correct"] = "N/A"
    
    return result
