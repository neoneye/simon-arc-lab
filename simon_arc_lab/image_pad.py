import numpy as np
import random

def image_pad_random(image: np.array, seed: int, color: int, min_pad_count: int, max_pad_count: int) -> np.array:
    """
    Draw a random sized border around the image.
    """
    if min_pad_count < 0:
        raise ValueError("min_pad_count must be at least 0.")
    if max_pad_count < min_pad_count:
        raise ValueError("max_pad_count must be at least as large as min_pad_count.")
    if max_pad_count > 256:
        raise ValueError("extremely big max_pad_count value.")
    top = random.Random(seed + 0).randint(min_pad_count, max_pad_count)
    bottom = random.Random(seed + 1).randint(min_pad_count, max_pad_count)
    left = random.Random(seed + 2).randint(min_pad_count, max_pad_count)
    right = random.Random(seed + 3).randint(min_pad_count, max_pad_count)
    image_padded = np.pad(image, ((top, bottom), (left, right)), mode='constant', constant_values=color)
    return image_padded
