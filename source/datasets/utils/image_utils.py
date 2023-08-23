import cv2
import numpy as np


def generate_image_binary_mask(image, channel_first=False):
    if channel_first:
        image = np.moveaxis(image, 0, -1)
    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 254, 1, cv2.THRESH_BINARY_INV)
    return binary_mask

def z_normalize_image(image, mean, std):
    assert image.shape[-1] == 3, "Image should be in channel last format"
    image = image.astype(np.float32)
    image -= mean
    image /= std
    return image