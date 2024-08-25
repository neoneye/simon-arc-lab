import numpy as np

def image_mask_and(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(image0)
    for y in range(image0.shape[0]):
        for x in range(image0.shape[1]):
            if image0[y, x] == 1 and image1[y, x] == 1:
                mask[y, x] = 1
    return mask

def image_mask_or(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(image0)
    for y in range(image0.shape[0]):
        for x in range(image0.shape[1]):
            if image0[y, x] == 1 or image1[y, x] == 1:
                mask[y, x] = 1
    return mask

def image_mask_xor(image0: np.ndarray, image1: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(image0)
    for y in range(image0.shape[0]):
        for x in range(image0.shape[1]):
            if image0[y, x] != image1[y, x]:
                mask[y, x] = 1
    return mask
