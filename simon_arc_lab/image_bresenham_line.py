import numpy as np

def image_bresenham_line(image: np.array, x0: int, y0: int, x1: int, y1: int, color: int) -> np.array:
    """
    Draw a line on an image using Bresenham's line algorithm.

    https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    :param image: The image to draw the line on
    :param x0: The x-coordinate of the start point
    :param y0: The y-coordinate of the start point
    :param x1: The x-coordinate of the end point
    :param y1: The y-coordinate of the end point
    :param color: The color to draw the line with
    :return: The image with the line drawn on it
    """
    height, width = image.shape

    # Check if the coordinates are outside the image bounds
    if not (0 <= x0 < width and 0 <= y0 < height and 0 <= x1 < width and 0 <= y1 < height):
        raise ValueError(f"Coordinates ({x0}, {y0}), ({x1}, {y1}) are outside the image bounds of width {width} and height {height}")

    # Clone the image to avoid mutating the original
    new_image = np.copy(image)

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        new_image[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return new_image
