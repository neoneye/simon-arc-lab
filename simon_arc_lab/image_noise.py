import numpy as np
import random

def image_noise_one_pixel(image: np.array, seed: int) -> np.array:
    """
    Add noise to the image by changing the color of one pixel.

    :param image: The original image
    :param seed: The seed for the random number generator
    :return: The noisy image
    """
    
    height, width = image.shape
    if width < 1 or height < 1:
        raise ValueError("Expected image size. width >= 1 and height >= 1.")

    x = random.Random(seed + 1).randint(0, width - 1)
    y = random.Random(seed + 2).randint(0, height - 1)

    # pick a color that is not the original color
    color = image[y, x]
    available_colors = list(range(10))
    available_colors.remove(color)
    new_color = random.Random(seed + 3).choice(available_colors)

    result_image = image.copy()
    result_image[y, x] = new_color
    return result_image
