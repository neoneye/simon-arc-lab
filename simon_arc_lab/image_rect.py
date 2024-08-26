import numpy as np
from .rectangle import Rectangle

def image_rect(image: np.array, rect: Rectangle, color: int) -> np.array:
    """
    Draw a filled rectangle on an image.

    :param image: The image to draw the rect on
    :param rect: The coordinates of the rectangle
    :param color: The color to filled with
    :return: The image with the rectangle drawn on it
    """
    height, width = image.shape
    this_image_rect = Rectangle(0, 0, width, height)
    intersection_rect = rect.intersection(this_image_rect)

    # Clone the image to avoid mutating the original
    new_image = np.copy(image)

    # Check if the coordinates are outside the image bounds
    if intersection_rect.is_empty():
        return np.copy(image)

    for y in range(intersection_rect.y, intersection_rect.y + intersection_rect.height):
        for x in range(intersection_rect.x, intersection_rect.x + intersection_rect.width):
            new_image[y, x] = color

    return new_image

def image_rect_hollow(image: np.array, rect: Rectangle, color: int, size: int) -> np.array:
    """
    Draw a hollow rectangle on an image.

    :param image: The image to draw the rect on
    :param rect: The coordinates of the rectangle
    :param color: The color to filled with
    :param size: The size of the rectangle border
    :return: The image with the rectangle drawn on it
    """
    height, width = image.shape
    this_image_rect = Rectangle(0, 0, width, height)
    intersection_rect = rect.intersection(this_image_rect)

    hollow_width = rect.width - size * 2
    hollow_height = rect.height - size * 2
    hollow_x = rect.x + size
    hollow_y = rect.y + size

    # Clone the image to avoid mutating the original
    new_image = np.copy(image)

    # Check if the coordinates are outside the image bounds
    if intersection_rect.is_empty():
        return np.copy(image)

    for y in range(intersection_rect.y, intersection_rect.y + intersection_rect.height):
        for x in range(intersection_rect.x, intersection_rect.x + intersection_rect.width):
            if x < hollow_x or x >= hollow_x + hollow_width or y < hollow_y or y >= hollow_y + hollow_height:
                new_image[y, x] = color

    return new_image
