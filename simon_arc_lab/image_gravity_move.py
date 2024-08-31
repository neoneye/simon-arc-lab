import numpy as np
from enum import Enum
from .image_util import *

class GravityMoveDirection(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

def image_gravity_move(image: np.array, background_color: int, direction: GravityMoveDirection) -> np.array:
    """
    Apply gravity to the image to the specified direction.

    :param image: The image to apply gravity to
    :param background_color: The color that is non-solid
    :param direction: The direction to apply gravity to
    :return: The result image with the gravity applied
    """
    if direction == GravityMoveDirection.LEFT:
        return _image_gravity_move_left(image, background_color)
    elif direction == GravityMoveDirection.RIGHT:
        image0 = np.fliplr(image)
        result_raw = _image_gravity_move_left(image0, background_color)
        return np.fliplr(result_raw)
    elif direction == GravityMoveDirection.UP:
        image0 = np.transpose(image)
        result_raw = _image_gravity_move_left(image0, background_color)
        return np.transpose(result_raw)
    elif direction == GravityMoveDirection.DOWN:
        image0 = np.fliplr(np.transpose(image))
        result_raw = _image_gravity_move_left(image0, background_color)
        return np.transpose(np.fliplr(result_raw))
    else:
        raise ValueError("Invalid direction")

def _image_gravity_move_left(image: np.array, background_color: int) -> np.array:
    """
    Apply gravity to the left direction.

    :param image: The image to apply gravity to
    :param background_color: The color that is non-solid
    :return: The result image with the gravity applied
    """
    height, width = image.shape

    new_image = np.full((height, width), background_color, dtype=np.uint8)
    for y in range(height):
        dest_x = 0
        for x in range(width):
            value = image[y, x]
            if value != background_color:
                new_image[y, dest_x] = value
                dest_x += 1

    return new_image

