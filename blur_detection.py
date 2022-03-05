import cv2
import numpy as np


def _variance_of_laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def get_blur_factor(img_pil):
    img_cv = np.array(img_pil)
    img_cv = img_cv[:, :, ::-1].copy()
    return int(_variance_of_laplacian(img_cv))
