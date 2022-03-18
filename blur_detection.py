import cv2
import numpy as np
from PIL import Image


def _variance_of_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def get_blur_factor(img_pil):
    img_cv = np.array(img_pil)
    img_cv = img_cv[:, :, ::-1].copy()
    return int(_variance_of_laplacian(img_cv))


def get_average_blur(imgs):
    summation = 0
    for img in imgs:
        img_pil = Image.fromarray(((img + 1) / 2.0 * 255).astype(np.uint8))
        summation += get_blur_factor(img_pil)

    return summation / len(imgs)
