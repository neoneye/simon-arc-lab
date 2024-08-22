import numpy as np

def probe_color_inner(pixel_list: list[int], edge_color: int) -> list[int]:
    """
    Determine the color of the previous color span.
    """
    new_pixel_list = [edge_color] * len(pixel_list)
    set_color = edge_color
    for x in range(1, len(pixel_list)):
        color_prev = pixel_list[x-1]
        color = pixel_list[x]
        if color_prev != color:
            set_color = color_prev                
        new_pixel_list[x] = set_color
    return new_pixel_list

def probe_color(pixel_list: list[int], edge_color: int, reverse: bool) -> list[int]:
    """
    Determine the color of the previous color span. Reverse the list if needed.
    """
    if reverse:
        return probe_color_inner(pixel_list[::-1], edge_color)[::-1]
    return probe_color_inner(pixel_list, edge_color)

def image_raytrace_probe_color_direction_up(image: np.array, edge_color: int) -> np.array:
    """
    Raytrace in the up-direction, and determine what color is there.
    """
    width = image.shape[1]
    new_image = np.zeros_like(image)
    for x in range(width):
        pixel_list = image[:, x].tolist()
        new_pixel_list = probe_color(pixel_list, edge_color, False)
        new_image[:, x] = new_pixel_list
    return new_image

def image_raytrace_probe_color_direction_down(image: np.array, edge_color: int) -> np.array:
    """
    Raytrace in the down-direction, and determine what color is there.
    """
    width = image.shape[1]
    new_image = np.zeros_like(image)
    for x in range(width):
        pixel_list = image[:, x].tolist()
        new_pixel_list = probe_color(pixel_list, edge_color, True)
        new_image[:, x] = new_pixel_list
    return new_image

def image_raytrace_probe_color_direction_left(image: np.array, edge_color: int) -> np.array:
    """
    Raytrace in the left-direction, and determine what color is there.
    """
    height = image.shape[0]
    new_image = np.zeros_like(image)
    for y in range(height):
        pixel_list = image[y].tolist()
        new_pixel_list = probe_color(pixel_list, edge_color, False)
        new_image[y] = new_pixel_list
    return new_image

def image_raytrace_probe_color_direction_right(image: np.array, edge_color: int) -> np.array:
    """
    Raytrace in the right-direction, and determine what color is there.
    """
    height = image.shape[0]
    new_image = np.zeros_like(image)
    for y in range(height):
        pixel_list = image[y].tolist()
        new_pixel_list = probe_color(pixel_list, edge_color, True)
        new_image[y] = new_pixel_list
    return new_image
