import numpy as np

def image_halfplane(image: np.array, x0: int, y0: int, x1: int, y1: int) -> np.array:
    """
    Fill the halfplane over the line given by (x0, y0) and (x1, y1).

    When x0, y0 is in the top-left corner and x1, y1 is in the bottom-right corner. Then the halfplane that gets filled out is the top-right halfplane.
    When x0, y0 is in the top-right corner and x1, y1 is in the bottom-left corner. Then the halfplane that gets filled out is the top-left halfplane.
    When x0, y0 is in the bottom-left corner and x1, y1 is in the top-right corner. Then the halfplane that gets filled out is the bottom-right halfplane.
    When x0, y0 is in the bottom-right corner and x1, y1 is in the top-left corner. Then the halfplane that gets filled out is the bottom-left halfplane.

    The x and y coordinates can be outside the image bounds.

    :param image: The image to draw on
    :param x0: The x-coordinate of the start point
    :param y0: The y-coordinate of the start point
    :param x1: The x-coordinate of the end point
    :param y1: The y-coordinate of the end point
    :return: The image with the drawing on it
    """
    height, width = image.shape

    draw_color = 1

    # Clone the image to avoid mutating the original
    new_image = np.copy(image)

    dx = x1 - x0
    dy = y1 - y0

    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2

    for y in range(height):
        for x in range(width):
            pos_x = x - mid_x
            pos_y = y - mid_y

            cross_product = pos_x * dy - pos_y * dx

            if cross_product > 0:
                new_image[y, x] = draw_color
                continue

    return new_image

