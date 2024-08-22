import numpy as np

def image_raytrace_probe_color_direction_up(image: np.array, edge_color: int) -> np.array:
    """
    Raytrace in the up-direction, and determine what color is there.
    """
    height, width = image.shape
    new_image = np.full((height, width), edge_color, dtype=np.uint8)
    for x in range(width):
        set_color = edge_color
        for y in range(1, height):
            color_prev = image[y-1, x]
            color = image[y, x]
            if color_prev != color:
                set_color = color_prev                
            new_image[y, x] = set_color
    return new_image

def image_raytrace_probe_color_direction_down(image: np.array, edge_color: int) -> np.array:
    """
    Raytrace in the down-direction, and determine what color is there.
    """
    height, width = image.shape
    new_image = np.full((height, width), edge_color, dtype=np.uint8)
    for x in range(width):
        set_color = edge_color
        for y_rev in range(1, height):
            y = height - y_rev - 1
            color_prev = image[y+1, x]
            color = image[y, x]
            if color_prev != color:
                set_color = color_prev                
            new_image[y, x] = set_color
    return new_image

def image_raytrace_probe_color_direction_left(image: np.array, edge_color: int) -> np.array:
    """
    Raytrace in the left-direction, and determine what color is there.
    """
    height, width = image.shape
    new_image = np.full((height, width), edge_color, dtype=np.uint8)
    for y in range(height):
        set_color = edge_color
        for x in range(1, width):
            color_prev = image[y, x-1]
            color = image[y, x]
            if color_prev != color:
                set_color = color_prev                
            new_image[y, x] = set_color
    return new_image

def image_raytrace_probe_color_direction_right(image: np.array, edge_color: int) -> np.array:
    """
    Raytrace in the right-direction, and determine what color is there.
    """
    height, width = image.shape
    new_image = np.full((height, width), edge_color, dtype=np.uint8)
    for y in range(height):
        set_color = edge_color
        for x_rev in range(1, width):
            x = width - x_rev - 1
            color_prev = image[y, x+1]
            color = image[y, x]
            if color_prev != color:
                set_color = color_prev                
            new_image[y, x] = set_color
    return new_image
