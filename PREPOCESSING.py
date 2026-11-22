
import cv2
import numpy as np

IMG_SIZE = 384


def preprocess_rgb(img, target_size=IMG_SIZE, return_geometry=False):
    geom = {
        "orig_shape": tuple(img.shape[:2]),
        "crop1": None,
        "crop2": None,
        "pad": (0, 0, 0, 0)
    }

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Remove outer black background
    _, thresh_outer = cv2.threshold(
        gray, 10, 255, cv2.THRESH_BINARY
    )
    contours, _ = cv2.findContours(
        thresh_outer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        img = img[y:y + h, x:x + w]
        geom["crop1"] = (x, y, w, h)

    original_area = img.shape[0] * img.shape[1]

    # Remove white borders safely
    gray2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask_retina = cv2.inRange(gray2, 15, 230)

    contours2, _ = cv2.findContours(
        mask_retina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours2:
        largest_contour2 = max(contours2, key=cv2.contourArea)
        x2, y2, w2, h2 = cv2.boundingRect(largest_contour2)

        if w2 * h2 >= 0.7 * original_area:
            img = img[y2:y2 + h2, x2:x2 + w2]
            geom["crop2"] = (x2, y2, w2, h2)

    # Square padding
    h, w = img.shape[:2]
    size = max(h, w)

    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left

    img = cv2.copyMakeBorder(
        img,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    geom["pad"] = (
        pad_top, pad_bottom, pad_left, pad_right
    )

    img = cv2.resize(img, (target_size, target_size))

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    img = img.astype(np.float32) / 255.0

    return (img, geom) if return_geometry else img
