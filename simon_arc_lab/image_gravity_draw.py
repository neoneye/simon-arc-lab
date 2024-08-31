import numpy as np
from enum import Enum
from .image_util import *

class GravityDrawDirection(Enum):
    TOP_TO_BOTTOM = 0
    BOTTOM_TO_TOP = 1
    LEFT_TO_RIGHT = 2
    RIGHT_TO_LEFT = 3

def image_gravity_draw(image: np.array, background_color: int, direction: GravityDrawDirection) -> np.array:
    """
    Apply gravity to the image to the specified direction.

    :param image: The image to apply gravity to
    :param background_color: The color that is non-solid
    :param direction: The direction to apply gravity to
    :return: The result image with the gravity applied
    """
    if direction == GravityDrawDirection.TOP_TO_BOTTOM:
        image0 = np.transpose(image)
        result_raw = _image_gravity_draw_right(image0, background_color)
        return np.transpose(result_raw)
    elif direction == GravityDrawDirection.BOTTOM_TO_TOP:
        image0 = np.fliplr(np.transpose(image))
        result_raw = _image_gravity_draw_right(image0, background_color)
        return np.transpose(np.fliplr(result_raw))
    elif direction == GravityDrawDirection.LEFT_TO_RIGHT:
        return _image_gravity_draw_right(image, background_color)
    elif direction == GravityDrawDirection.RIGHT_TO_LEFT:
        image0 = np.fliplr(image)
        result_raw = _image_gravity_draw_right(image0, background_color)
        return np.fliplr(result_raw)
    else:
        raise ValueError("Invalid direction")

def _image_gravity_draw_right(image: np.array, background_color: int) -> np.array:
    """
    Apply gravity to the right direction.

    :param image: The image to apply gravity to
    :param background_color: The color that is non-solid
    :return: The result image with the gravity applied
    """
    height, width = image.shape

    new_image = image.copy()
    for y in range(height):
        set_color = background_color
        for x in range(width):
            value = image[y, x]
            if value == background_color:
                new_image[y, x] = set_color
            else:
                set_color = value

    return new_image

