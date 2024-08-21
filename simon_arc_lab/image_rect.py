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
