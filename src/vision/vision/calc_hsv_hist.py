#!/usr/bin/env python3
"""Helper script to preview polygon ROIs on a reference image and save HSV hue histograms per color."""

import argparse
from pathlib import Path
import cv2
import numpy as np

current_directory = Path.cwd()

# rois = {
#     "green":(
#         [56, 37],
#         [128, 37],
#         [128, 138],
#         [56, 138]),
#     "red": (
#         [212, 40],
#         [275, 40],
#         [275, 130],
#         [212, 130],
#     ),
#     "blue": (
#         [350, 35],
#         [410, 35],
#         [410, 130],
#         [350, 130],
#     )
# }

# # only front FACE
# rois = {
#     "green":(
#         [55, 80],
#         [125, 80],
#         [125, 140],
#         [55, 140]),
#     "red": (
#         [205, 80],
#         [283, 80],
#         [283, 142],
#         [205, 142],
#     ),
#     "blue": (
#         [343, 79],
#         [422, 79],
#         [422, 135],
#         [343, 135],
#     )
# }

# # only front FACE new image
# rois = {
#     "green":(
#         [452, 70],
#         [525, 70],
#         [525, 128],
#         [452, 128]),
#     "red": (
#         [224, 81],
#         [308, 81],
#         [308, 150],
#         [224, 150],
#     ),
#     "blue": (
#         [38, 90],
#         [115, 90],
#         [115, 150],
#         [38, 150],
#     )
# }

# only front FACE new image
rois = {
    "green":(
        [139, 396],
        [210, 396],
        [210, 435],
        [139, 435]),
    "red": (
        [308, 401],
        [394, 401],
        [394, 436],
        [308, 436],
    ),
    "blue": (
        [500, 387],
        [547, 387],
        [547, 414],
        [500, 414],
    )
}

def load_img(path):
    """Load an image from disk (BGR) or raise if missing."""
    img = cv2.imread(path)  # BGR
    if img is None:
        raise FileNotFoundError("Image not found")
    return img

def show_img(img):
    """Display an image and block until a key is pressed."""
    cv2.imshow("Mask preview (press space)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_cropped_polygon_roi(img, roi_pts, win="ROI crop"):
    """Show the masked polygon ROI cropped to its bounding rectangle."""
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
    """Convert a BGR image to HSV."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def comput_hist(roi_dict, img, save=False):
    """Compute and optionally save normalized hue histograms for each polygon ROI."""
    hsv = bgr_to_hsv(img=img)

    for i, (color, roi) in enumerate(roi_dict.items()):
        if not roi:
            print(f"Skipping {color}: no ROI points")
            continue

        print(i, color, "\n", roi, "\n")

        roi = np.array(roi, dtype=np.int32)

        mask_poly = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask_poly, roi, 255)

        # Optional: additionally mask low S/V for robustness against shadows/glare.
        # mask_sv = cv2.inRange(hsv, (0, 40, 40), (179, 255, 255))
        # mask = cv2.bitwise_and(mask_poly, mask_sv)

        hist = cv2.calcHist([hsv], [0], mask_poly, [60], [0, 180])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        if save:
            out = Path.joinpath(current_directory, f"src/vision/histograms/hist_{color}_front-face_new2.npy") 
            np.save(out, hist)
            print(f"Gespeichert: hist_{color}_front-face_new2.npy")

        print("save?", save)

def main():
    """Run an interactive ROI preview and export histograms from a fixed reference image."""
    path = Path.joinpath(current_directory, "imgs/hsv_reference_3.png")     # TODO adapt path
    print("Loading image from:", path)
    img = load_img(str(path))
    show_img(img)

    show_cropped_polygon_roi(img=img, roi_pts=rois["green"])
    show_cropped_polygon_roi(img=img, roi_pts=rois["blue"])
    show_cropped_polygon_roi(img=img, roi_pts=rois["red"])
    comput_hist(rois, img, save=True)

if __name__ == "__main__":
    main()
