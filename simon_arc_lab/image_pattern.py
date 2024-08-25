import numpy as np

def image_pattern_checkerboard(width: int, height: int, square_size: int, offsetx: int, offsety: int, colors: list[int]) -> np.array:
    """
    Create a checkerboard pattern image.

    :param width: The width of the image
    :param height: The height of the image
    :param square_size: The size of each square in the checkerboard
    :param offsetx: Displacement in x direction
    :param offsety: Displacement in y direction
    :param colors: The color of the squares
    :return: The checkerboard pattern image
    """
    image = np.zeros((height, width), dtype=np.uint8)
    color_count = len(colors)
    for y in range(height):
        for x in range(width):
            cell_x = (x + offsetx) // square_size
            cell_y = (y + offsety) // square_size
            color_index = (cell_x + cell_y) % color_count
            image[y, x] = colors[color_index]
    return image

def image_pattern_lines_horizontal(width: int, height: int, line_size: int, offsety: int, colors: list[int]) -> np.array:
    """
    Create a pattern with repeating lines.

    :param width: The width of the image
    :param height: The height of the image
    :param line_size: The size of each line
    :param offsety: Displacement in y direction
    :param colors: The color of the lines
    :return: The generated pattern image
    """
    image = np.zeros((height, width), dtype=np.uint8)
    color_count = len(colors)
    for y in range(height):
        for x in range(width):
            cell_y = (y + offsety) // line_size
            color_index = cell_y % color_count
            image[y, x] = colors[color_index]
    return image

def image_pattern_lines_slope(width: int, height: int, dx: int, dy: int, colors: list[int]) -> np.array:
    """
    Create a pattern with lines at a given slope.

    :param width: The width of the image
    :param height: The height of the image
    :param dx: Move dx pixels in x direction
    :param dy: Move dy pixels in y direction
    :param colors: The color of the lines
    :return: The generated pattern image
    """
    image = np.zeros((height, width), dtype=np.uint8)
    color_count = len(colors)
    
    # Normalize the slope
    gcd = np.gcd(dx, dy)
    step_x = np.abs(dx) // gcd
    step_y = np.abs(dy) // gcd
    
    for y in range(height):
        for x in range(width):            
            cell_index = (x // step_x) + (y // step_y)
            image[y, x] = colors[cell_index % color_count]

    return image
