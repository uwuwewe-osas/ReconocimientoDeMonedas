import glob
import os

imagenes = glob.glob('MonedasHallar/*.jpg')
imagenes.sort(key=lambda x: int(os.path.basename(x).replace('monedas (', '').replace(').jpg', '')))

with open('valores_reales.txt', 'w', encoding='utf-8') as f:
    for img_path in imagenes:
        nombre = os.path.basename(img_path)
        f.write(f"{nombre}\n")
        f.write("10 centimos: 0\n")
        f.write("20 centimos: 0\n")
        f.write("50 centimos: 0\n")
        f.write("1 sol: 0\n")
        f.write("2 soles: 0\n")
        f.write("5 soles: 0\n")
        f.write("TOTAL REAL: 0.00\n\n")

print("Generado valores_reales.txt")
