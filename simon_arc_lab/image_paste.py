import numpy as np
import random
from .rectangle import Rectangle

def rectangle_for_random_paste(paste_image: np.array, background_image: np.array, seed: int) -> Rectangle:
    """
    Pick a random position where an image can be pasted, so it fits inside another image.

    The paste_image must fit inside the background_image.
    If it's too big an exception is raised.

    :param paste_image: The image to insert.
    :param background_image: The image to insert into.
    :param seed: The seed for the random number generator.
    :return: The rectangle where the paste_image can be inserted.
    """
    paste_height, paste_width = paste_image.shape
    background_height, background_width = background_image.shape
    if paste_height > background_height or paste_width > background_width:
        raise ValueError("The paste image must fit inside the background image.")
    available_height = background_height - paste_height
    available_width = background_width - paste_width

    top = random.Random(seed + 0).randint(0, available_height)
    left = random.Random(seed + 1).randint(0, available_width)
    return Rectangle(left, top, paste_width, paste_height)

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
    rectangle = rectangle_for_random_paste(paste_image, background_image, seed)
    return image_paste_at(paste_image, background_image, rectangle.x, rectangle.y)

def image_paste_at(paste_image: np.array, background_image: np.array, x: int, y: int) -> np.array:
    """
    Paste an image at the x,y position inside another image.

    The paste_image must fit inside the background_image.
    If it's too big an exception is raised.

    The x,y position must be valid for the paste_image to fit inside the background_image.
    If the image goes outside the background_image an exception is raised.

    :param paste_image: The image to insert.
    :param background_image: The image to insert into.
    :param x: The x position.
    :param y: The y position.
    :return: A new image with the paste_image inserted.
    """
    paste_height, paste_width = paste_image.shape
    background_height, background_width = background_image.shape
    x0 = x
    x1 = x0 + paste_width
    y0 = y
    y1 = y0 + paste_height
    if x0 < 0 or y0 < 0 or x1 > background_width or y1 > background_height:
        raise ValueError("The paste image must fit inside the background image")
    image = background_image.copy()
    image[y0:y1, x0:x1] = paste_image
    return image
