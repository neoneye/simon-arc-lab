import numpy as np
import random

def image_paste_random(paste_image: np.array, background_image: np.array, seed: int) -> np.array:
    """
    Paste an image at a random position inside another image.

    The paste_image must fit inside the background_image.
    If it's too big an exception is raised.

    :param paste_image: The image to insert.
    :param background_image: The image to insert into.
    :param seed: The seed for the random number generator.
    :return: A new image with the paste_image inserted.
    """
    paste_height, paste_width = paste_image.shape
    background_height, background_width = background_image.shape
    if paste_height > background_height or paste_width > background_width:
        raise ValueError("The paste image must fit inside the background image.")
    available_height = background_height - paste_height
    available_width = background_width - paste_width

    top = random.Random(seed + 0).randint(0, available_height)
    left = random.Random(seed + 1).randint(0, available_width)
    image = background_image.copy()
    image[top:top + paste_height, left:left + paste_width] = paste_image
    return image
