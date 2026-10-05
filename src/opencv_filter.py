import cv2
import numpy as np


def gaussian_blur_opencv(image: np.ndarray, size: int, sigma: float) -> np.ndarray:
    """Размытие изображения встроенной функцией cv2.GaussianBlur"""
    return cv2.GaussianBlur(
        image,
        ksize=(size, size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT,
    )