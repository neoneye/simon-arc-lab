import numpy as np

def image_mix(mask: np.array, image0: np.array, image1: np.array) -> np.array:
    """
    Mix two images with a mask.

    Where the mask is 0, the pixel is taken from image0.
    Where the mask is non-zero, the pixel is taken from image1.
    """
    return np.where(mask, image1, image0)
