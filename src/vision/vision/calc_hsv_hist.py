#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np

rois = {
    "green":( 
    [56, 37],
    [128, 37],
    [128, 138],
    [56, 138]),
    "red": (
     [212, 40],   
     [275, 40],   
     [275, 130],   
     [212, 130],   
    ),
    "blue": (
     [350, 35],   
     [410, 35],   
     [410, 130],   
     [350, 130],
    )   
}

"""roi_green=np.array([
    [56, 37],
    [128, 37],
    [56, 138],
    [128, 138]
], dtype=np.int32)"""

def load_img(path):
    img = cv2.imread(path)  # BGR
    if img is None:
        raise FileNotFoundError("Bild nicht gefunden")
    return img

def show_img(img):
    cv2.imshow("Mask preview (press any key)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_cropped_polygon_roi(img, roi_pts, win="ROI crop"):
    roi = np.array(roi_pts, dtype=np.int32)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, roi, 255)

    x, y, w, h = cv2.boundingRect(roi)
    cropped = img[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    cropped_roi = cv2.bitwise_and(cropped, cropped, mask=cropped_mask)

    cv2.imshow(win, cropped_roi)
    cv2.waitKey(0)
    cv2.destroyWindow(win)

def bgr_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def comput_hist(roi_dict, img, save=False):
    hsv = bgr_to_hsv(img=img)

    for i, (color, roi) in enumerate(roi_dict.items()):
        if not roi:  # empty tuple for red/blue right now
            print(f"Skipping {color}: no ROI points")
            continue

        print(i, color, "\n", roi, "\n")
        
        roi = np.array(roi, dtype=np.int32)

        mask_poly = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask_poly, roi, 255)


        """    # Optional: zusätzlich low S/V rausmaskieren (robuster gegen Schatten/Glanz)
        mask_sv = cv2.inRange(hsv, (0, 40, 40), (179, 255, 255))
        mask = cv2.bitwise_and(mask_poly, mask_sv)"""

        # --- 4) Hue-Histogramm berechnen (tolerant: z.B. 60 Bins statt 180) ---
        hist = cv2.calcHist([hsv], [0], mask_poly, [60], [0, 180])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        # --- 5) Speichern ---
        if save:
            out=f"/home/maalonjochmann/HRS_ST3AL/src/vision/vision/maalon/hist_{color}.npy"
            np.save(out, hist)

            print(f"Gespeichert: hist_{color}.npy")
        
        print("save?", save)

def main():
    path ="/home/maalonjochmann/HRS_ST3AL/imgs/hsv_reference.jpg"
    img = load_img(path)
    show_img(img)

    #show_cropped_polygon_roi(img=img, roi_pts=rois["green"])
    show_cropped_polygon_roi(img=img, roi_pts=rois["blue"])
    show_cropped_polygon_roi(img=img, roi_pts=rois["red"])
    comput_hist(rois, img, save=True)

if __name__ == "__main__":
    main()

