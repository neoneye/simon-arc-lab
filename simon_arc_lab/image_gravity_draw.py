import numpy as np
from enum import Enum
from .image_util import *
from .image_skew import *

GRAVITY_DRAW_SKEW_PADDING_COLOR = 255

class GravityDrawDirection(Enum):
    TOP_TO_BOTTOM = 0
    BOTTOM_TO_TOP = 1
    LEFT_TO_RIGHT = 2
    RIGHT_TO_LEFT = 3
    TOPLEFT_TO_BOTTOMRIGHT = 4
    BOTTOMRIGHT_TO_TOPLEFT = 5
    TOPRIGHT_TO_BOTTOMLEFT = 6
    BOTTOMLEFT_TO_TOPRIGHT = 7

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
    elif direction == GravityDrawDirection.TOPLEFT_TO_BOTTOMRIGHT:
        return _image_gravity_draw_topleft_to_bottomright(image, background_color)
    elif direction == GravityDrawDirection.TOPRIGHT_TO_BOTTOMLEFT:
        image0 = np.fliplr(image)
        result_raw = _image_gravity_draw_topleft_to_bottomright(image0, background_color)
        return np.fliplr(result_raw)
    elif direction == GravityDrawDirection.BOTTOMLEFT_TO_TOPRIGHT:
        # flip both horizontally and vertically
        image0 = np.flipud(image)
        result_raw = _image_gravity_draw_topleft_to_bottomright(image0, background_color)
        return np.flipud(result_raw)
    elif direction == GravityDrawDirection.BOTTOMRIGHT_TO_TOPLEFT:
        # flip both horizontally and vertically
        image0 = np.flip(image, axis=(0, 1))
        result_raw = _image_gravity_draw_topleft_to_bottomright(image0, background_color)
        return np.flip(result_raw, axis=(0, 1))
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
            if value == GRAVITY_DRAW_SKEW_PADDING_COLOR:
                # Ignore the area outside the skewed image
                continue
            if value == background_color:
                new_image[y, x] = set_color
            else:
                set_color = value

    return new_image

def _image_gravity_draw_topleft_to_bottomright(image: np.array, background_color: int) -> np.array:
    """
    Apply directional gravity from topleft to bottomright.

    :param image: The image to apply gravity to
    :param background_color: The color that is non-solid
    :return: The result image with the gravity applied
    """
    skewed_image = image_skew_up(image, GRAVITY_DRAW_SKEW_PADDING_COLOR)
    transformed_image = _image_gravity_draw_right(skewed_image, background_color)
    return image_unskew_up(transformed_image)
