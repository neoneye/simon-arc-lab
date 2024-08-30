import numpy as np
from enum import Enum
from .image_util import *

class ReverseDirection(Enum):
    TOPBOTTOM = 0
    LEFTRIGHT = 1

def image_reverse(image: np.array, split_color: int, direction: ReverseDirection) -> np.array:
    """
    Reverse chunks of pixels in the specified direction.

    :param image: The image to transform
    :param split_color: The color that splits the chunks
    :param direction: The direction
    :return: The transformed image
    """
    if direction == ReverseDirection.LEFTRIGHT:
        return _image_reverse_horizontal(image, split_color)
    elif direction == ReverseDirection.TOPBOTTOM:
        image0 = np.transpose(image)
        result_raw = _image_reverse_horizontal(image0, split_color)
        return np.transpose(result_raw)
    else:
        raise ValueError("Invalid direction")

def _image_reverse_horizontal(image: np.array, split_color: int) -> np.array:
    """
    Reverse chunks of pixels in the horizontal direction.

    :param image: The image to transform
    :param split_color: The color that splits the chunks
    :return: The transformed image
    """
    height, width = image.shape

    new_image = np.full((height, width), split_color, dtype=np.uint8)
    for y in range(height):
        start_x = None
        count = 0
        for x in range(width):
            if image[y, x] == split_color:
                continue
            if start_x is not None:
                count += 1
            else:
                start_x = x
                count = 1

            if x == width - 1 or image[y, x + 1] == split_color:
                for i in range(count):
                    new_image[y, start_x + count - i - 1] = image[y, start_x + i]
                start_x = None
                count = 0

    return new_image

