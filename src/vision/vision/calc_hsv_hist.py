import cv2
import numpy as np

img = cv2.imread("green_ref.png")  # BGR
if img is None:
    raise FileNotFoundError("Bild nicht gefunden")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# --- 2) Eckpunkte der Würfelfläche (Pixelkoordinaten) ---
# Beispiel: PUNKTE ANPASSEN!
pts = np.array([
    [640, 560],
    [730, 540],
    [750, 630],
    [655, 650],
], dtype=np.int32)

# --- 3) Polygon-Maske erstellen ---
mask_poly = np.zeros(hsv.shape[:2], dtype=np.uint8)
cv2.fillConvexPoly(mask_poly, pts, 255)

# Optional: zusätzlich low S/V rausmaskieren (robuster gegen Schatten/Glanz)
mask_sv = cv2.inRange(hsv, (0, 40, 40), (179, 255, 255))
mask = cv2.bitwise_and(mask_poly, mask_sv)

# --- 4) Hue-Histogramm berechnen (tolerant: z.B. 60 Bins statt 180) ---
hist = cv2.calcHist([hsv], [0], mask, [60], [0, 180])
cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

# --- 5) Speichern ---
np.save("hist_green.npy", hist)

print("Gespeichert: hist_green.npy")
