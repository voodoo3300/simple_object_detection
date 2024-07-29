import cv2
import numpy as np
import os

# Verzeichnisse
input_dir = 'images'
output_dir = 'out'

# Erstellen des Ausgabeordners, falls nicht vorhanden
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Parameter
scale_percent = 20  # Prozent des Originals, anpassen je nach Bedarf
padding = 25  # Rahmenvergrößerung

# Schleife durch alle Dateien im Eingabeverzeichnis
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Bildpfad
        image_path = os.path.join(input_dir, filename)

        # Bild laden
        image_orig = cv2.imread(image_path)

        # Bild verkleinern
        width = int(image_orig.shape[1] * scale_percent / 100)
        height = int(image_orig.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image_orig, dim, interpolation=cv2.INTER_AREA)

        # Bild in Graustufen umwandeln
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gaussian Blur anwenden, um das Bild zu glätten und Rauschen zu reduzieren
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # Canny-Kantenerkennung durchführen
        edges = cv2.Canny(blurred, 50, 150)

        # Morphologische Operationen, um kleine Störungen zu entfernen und Konturen zu verbinden
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=2)

        # Konturen finden
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Schleife durch alle gefundenen Konturen und nach Größe filtern
        filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

        for i, contour in enumerate(filtered_contours):
            # Bounding Box für jede Kontur berechnen
            x, y, w, h = cv2.boundingRect(contour)

            # Skalieren der Bounding Box Koordinaten auf das Originalbild
            x_orig = int(x / scale_percent * 100) - padding
            y_orig = int(y / scale_percent * 100) - padding
            w_orig = int(w / scale_percent * 100) + 2 * padding
            h_orig = int(h / scale_percent * 100) + 2 * padding

            # Begrenzungen sicherstellen, dass sie innerhalb des Bildes bleiben
            x_orig = max(0, x_orig)
            y_orig = max(0, y_orig)
            w_orig = min(image_orig.shape[1] - x_orig, w_orig)
            h_orig = min(image_orig.shape[0] - y_orig, h_orig)

            # Objekt aus dem Originalbild ausschneiden
            cropped = image_orig[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]

            # Ausgeschnittenes Objekt speichern
            output_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_{i}.png')
            cv2.imwrite(output_path, cropped)

        print(f"Relevante Objekte aus {filename} wurden erkannt und gespeichert.")

print("Alle Bilder wurden verarbeitet.")
