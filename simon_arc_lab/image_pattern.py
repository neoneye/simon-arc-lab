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
    return image_pattern_lines_slope_advanced(width, height, 1, 1, square_size, offsetx, offsety, colors)

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
    return image_pattern_lines_slope_advanced(width, height, 0, 1, line_size, 0, offsety, colors)

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
    return image_pattern_lines_slope_advanced(width, height, dx, dy, 1, 0, 0, colors)

def image_pattern_lines_slope_advanced(width: int, height: int, dx: int, dy: int, square_size: int, offsetx: int, offsety: int, colors: list[int]) -> np.array:
    """
    Create a pattern with lines at a given slope.

    :param width: The width of the image
    :param height: The height of the image
    :param dx: Move dx pixels in x direction
    :param dy: Move dy pixels in y direction
    :param square_size: The size of each square in the output image
    :param offsetx: Displacement in x direction
    :param offsety: Displacement in y direction
    :param colors: The color of the lines
    :return: The generated pattern image
    """
    image = np.zeros((height, width), dtype=np.uint8)
    color_count = len(colors)
    
    if dx == 0 and dy == 0:
        raise ValueError("Both dx and dy must not be zero")
    if dx != 0 and dy != 0:
        # Normalize the slope
        gcd = np.gcd(dx, dy)
        step_x = np.abs(dx) // gcd
        step_y = np.abs(dy) // gcd
    if dx == 0:
        step_x = 0
        step_y = dy // np.abs(dy) # Either 1 or -1
    if dy == 0:
        step_x = dx // np.abs(dx) # Either 1 or -1
        step_y = 0

    for y in range(height):
        for x in range(width):
            cell_index = 0
            if step_x != 0:
                cell_x = (x + offsetx) // square_size
                cell_index += (cell_x // step_x)
            if step_y != 0:
                cell_y = (y + offsety) // square_size
                cell_index += (cell_y // step_y)
            image[y, x] = colors[cell_index % color_count]

    return image
