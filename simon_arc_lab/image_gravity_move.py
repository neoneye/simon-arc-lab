import numpy as np
from enum import Enum
from .image_util import *
from .image_skew import *

GRAVITY_MOVE_SKEW_PADDING_COLOR = 255

class GravityMoveDirection(Enum):
    TOP_TO_BOTTOM = 0
    BOTTOM_TO_TOP = 1
    LEFT_TO_RIGHT = 2
    RIGHT_TO_LEFT = 3
    TOPLEFT_TO_BOTTOMRIGHT = 4
    BOTTOMRIGHT_TO_TOPLEFT = 5
    TOPRIGHT_TO_BOTTOMLEFT = 6
    BOTTOMLEFT_TO_TOPRIGHT = 7

def image_gravity_move(image: np.array, background_color: int, direction: GravityMoveDirection) -> np.array:
    """
    Apply gravity to the image to the specified direction.

    :param image: The image to apply gravity to
    :param background_color: The color that is non-solid
    :param direction: The direction to apply gravity to
    :return: The result image with the gravity applied
    """
    if direction == GravityMoveDirection.TOP_TO_BOTTOM:
        image0 = np.fliplr(np.transpose(image))
        result_raw = _image_gravity_move_left(image0, background_color)
        return np.transpose(np.fliplr(result_raw))
    elif direction == GravityMoveDirection.BOTTOM_TO_TOP:
        image0 = np.transpose(image)
        result_raw = _image_gravity_move_left(image0, background_color)
        return np.transpose(result_raw)
    elif direction == GravityMoveDirection.LEFT_TO_RIGHT:
        image0 = np.fliplr(image)
        result_raw = _image_gravity_move_left(image0, background_color)
        return np.fliplr(result_raw)
    elif direction == GravityMoveDirection.RIGHT_TO_LEFT:
        return _image_gravity_move_left(image, background_color)
    elif direction == GravityMoveDirection.TOPLEFT_TO_BOTTOMRIGHT:
        # flip both horizontally and vertically
        image0 = np.flip(image, axis=(0, 1))
        result_raw = _image_gravity_move_bottomright_to_topleft(image0, background_color)
        return np.flip(result_raw, axis=(0, 1))
    elif direction == GravityMoveDirection.BOTTOMRIGHT_TO_TOPLEFT:
        return _image_gravity_move_bottomright_to_topleft(image, background_color)
    elif direction == GravityMoveDirection.TOPRIGHT_TO_BOTTOMLEFT:
        image0 = np.flipud(image)
        result_raw = _image_gravity_move_bottomright_to_topleft(image0, background_color)
        return np.flipud(result_raw)
    elif direction == GravityMoveDirection.BOTTOMLEFT_TO_TOPRIGHT:
        image0 = np.fliplr(image)
        result_raw = _image_gravity_move_bottomright_to_topleft(image0, background_color)
        return np.fliplr(result_raw)
    else:
        raise ValueError("Invalid direction")

def _image_gravity_move_left(image: np.array, background_color: int) -> np.array:
    """
    Apply directional gravity from left to right.

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
            if value == GRAVITY_MOVE_SKEW_PADDING_COLOR:
                # Ignore the area outside the skewed image
                dest_x += 1
                continue
            if value != background_color:
                new_image[y, dest_x] = value
                dest_x += 1

    return new_image

def _image_gravity_move_bottomright_to_topleft(image: np.array, background_color: int) -> np.array:
    """
    Apply directional gravity from bottomright to topleft.

    :param image: The image to apply gravity to
    :param background_color: The color that is non-solid
    :return: The result image with the gravity applied
    """
    skewed_image = image_skew(image, GRAVITY_MOVE_SKEW_PADDING_COLOR, SkewDirection.UP)
    transformed_image = _image_gravity_move_left(skewed_image, background_color)
    return image_unskew(transformed_image, SkewDirection.UP)
