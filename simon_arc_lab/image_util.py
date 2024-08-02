from typing import Dict
import numpy as np

def image_create(width: int, height: int, color: int) -> np.array:
    image = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image[y, x] = color
    return image

def image_rotate_cw(image: np.array) -> np.array:
    return np.rot90(image, k=-1)

def image_rotate_ccw(image: np.array) -> np.array:
    return np.rot90(image)

def image_rotate_180(image: np.array) -> np.array:
    return np.rot90(image, k=2)

def image_flipx(image: np.array) -> np.array:
    """
    Flip an image horizontally. Reverse the x-axis.
    """
    return np.fliplr(image)

def image_flipy(image: np.array) -> np.array:
    """
    Flip an image vertically. Reverse the y-axis.
    """
    return np.flipud(image)

def image_translate_wrap(image: np.array, dx: int, dy: int) -> np.array:
    """
    Move pixels by dx, dy, wrapping around the image.

    :param image: The image to process.
    :param dx: The horizontal translation.
    :param dy: The vertical translation.
    :return: An image of the same size as the input image.
    """
    if dx == 0 and dy == 0:
        raise ValueError("dx and dy cannot both be zero.")

    height, width = image.shape
    new_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            new_y = (height + y + dy) % height
            new_x = (width + x + dx) % width
            new_image[new_y, new_x] = image[y, x]

    return new_image

def image_replace_colors(image: np.array, color_mapping: Dict[int, int]) -> np.array:
    """
    Replace colors in an image according to a dictionary.

    :param image: The image to process.
    :param color_mapping: A dictionary where the keys are the colors to replace and the values are the new colors.
    :return: An image of the same size as the input image.
    """
    new_image = np.copy(image)
    
    for old_color, new_color in color_mapping.items():
        mask = image == old_color
        new_image[mask] = new_color
        
    return new_image

def image_get_row_as_list(image: np.array, row_index: int) -> list[int]:
    """
    Get a row from an image.

    :param image: The image to process.
    :param row_index: The index of the row to get.
    :return: The row as a list.
    """
    height = image.shape[0]
    if row_index < 0 or row_index >= height:
        raise ValueError(f"Row index {row_index} is out of bounds for image with height {height}")
    return list(image[row_index])

def image_get_column_as_list(image: np.array, column_index: int) -> list[int]:
    """
    Get a column from an image.

    :param image: The image to process.
    :param column_index: The index of the column to get.
    :return: The column as a list.
    """
    width = image.shape[1]
    if column_index < 0 or column_index >= width:
        raise ValueError(f"Column index {column_index} is out of bounds for image with width {width}")
    return list(image[:, column_index])
