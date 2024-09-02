import numpy as np
from enum import Enum

class SkewDirection(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

def image_skew(image: np.array, padding_color: int, direction: SkewDirection) -> np.array:
    """
    Skew the image in the specified direction.

    :param image: The image to skew
    :param padding_color: The color to use for padding
    :param direction: The direction to skew the image
    :return: The skewed image
    """
    if direction == SkewDirection.UP:
        return _image_skew_up(image, padding_color)
    elif direction == SkewDirection.DOWN:
        return _image_skew_down(image, padding_color)
    elif direction == SkewDirection.LEFT:
        return _image_skew_left(image, padding_color)
    elif direction == SkewDirection.RIGHT:
        return _image_skew_right(image, padding_color)
    else:
        raise ValueError("Invalid direction")

def image_unskew(image: np.array, direction: SkewDirection) -> np.array:
    """
    Unskew the image in the specified direction.

    :param image: The image to unskew
    :param direction: The direction to unskew the image
    :return: The unskewed image
    """
    if direction == SkewDirection.UP:
        return _image_unskew_up(image)
    elif direction == SkewDirection.DOWN:
        return _image_unskew_down(image)
    elif direction == SkewDirection.LEFT:
        return _image_unskew_left(image)
    elif direction == SkewDirection.RIGHT:
        return _image_unskew_right(image)
    else:
        raise ValueError("Invalid direction")

def _image_skew_up(image: np.array, padding_color: int) -> np.array:
    """
    Displace each column up by the column index.
    """
    height, width = image.shape
    skewed_image = np.full((height + width - 1, width), padding_color, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            skewed_image[width-1-x+y, x] = image[y, x]
    return skewed_image

def _image_unskew_up(image: np.array) -> np.array:
    """
    Displace each column down by the column index. Remove the padding.
    """
    height, width = image.shape
    height_unskewed = height - width + 1
    unskewed_image = np.zeros((height_unskewed, width), dtype=np.uint8)
    for y in range(height_unskewed):
        for x in range(width):
            unskewed_image[y, x] = image[width-1-x+y, x]
    return unskewed_image

def _image_skew_down(image: np.array, padding_color: int) -> np.array:
    """
    Displace each column down by the column index.
    """
    height, width = image.shape
    skewed_image = np.full((height + width - 1, width), padding_color, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            skewed_image[y+x, x] = image[y, x]
    return skewed_image

def _image_unskew_down(image: np.array) -> np.array:
    """
    Displace each column up by the column index. Remove the padding.
    """
    height, width = image.shape
    height_unskewed = height - width + 1
    unskewed_image = np.zeros((height_unskewed, width), dtype=np.uint8)
    for y in range(height_unskewed):
        for x in range(width):
            unskewed_image[y, x] = image[y+x, x]
    return unskewed_image

def _image_skew_left(image: np.array, padding_color: int) -> np.array:
    """
    Displace each row to the left by the row index.
    """
    height, width = image.shape
    skewed_image = np.full((height, height + width - 1), padding_color, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            skewed_image[y, height-1-y+x] = image[y, x]
    return skewed_image

def _image_unskew_left(image: np.array) -> np.array:
    """
    Displace each row to the right by the row index. Remove the padding.
    """
    height, width = image.shape
    width_unskewed = width - height + 1
    unskewed_image = np.zeros((height, width_unskewed), dtype=np.uint8)
    for y in range(height):
        for x in range(width_unskewed):
            unskewed_image[y, x] = image[y, height-1-y+x]
    return unskewed_image

def _image_skew_right(image: np.array, padding_color: int) -> np.array:
    """
    Displace each row to the right by the row index.
    """
    height, width = image.shape
    skewed_image = np.full((height, height + width - 1), padding_color, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            skewed_image[y, y+x] = image[y, x]
    return skewed_image

def _image_unskew_right(image: np.array) -> np.array:
    """
    Displace each row to the left by the row index. Remove the padding.
    """
    height, width = image.shape
    width_unskewed = width - height + 1
    unskewed_image = np.zeros((height, width_unskewed), dtype=np.uint8)
    for y in range(height):
        for x in range(width_unskewed):
            unskewed_image[y, x] = image[y, y+x]
    return unskewed_image
