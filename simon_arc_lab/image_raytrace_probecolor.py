import numpy as np
from enum import Enum
from .image_skew import *

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

def image_raytrace_probecolor_direction_top(image: np.array, edge_color: int) -> np.array:
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

def image_raytrace_probecolor_direction_bottom(image: np.array, edge_color: int) -> np.array:
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

def image_raytrace_probecolor_direction_left(image: np.array, edge_color: int) -> np.array:
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

def image_raytrace_probecolor_direction_right(image: np.array, edge_color: int) -> np.array:
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

def image_raytrace_probecolor_direction_topleft(image: np.array, edge_color: int) -> np.array:
    """
    Raytrace in the upleft-direction, and determine what color is there.
    """
    skewed_image = image_skew(image, edge_color, SkewDirection.LEFT)
    skewed_image = image_raytrace_probecolor_direction_top(skewed_image, edge_color)
    unskewed_image = image_unskew(skewed_image, SkewDirection.LEFT)
    return unskewed_image

def image_raytrace_probecolor_direction_topright(image: np.array, edge_color: int) -> np.array:
    """
    Raytrace in the upright-direction, and determine what color is there.
    """
    skewed_image = image_skew(image, edge_color, SkewDirection.RIGHT)
    skewed_image = image_raytrace_probecolor_direction_top(skewed_image, edge_color)
    unskewed_image = image_unskew(skewed_image, SkewDirection.RIGHT)
    return unskewed_image

def image_raytrace_probecolor_direction_bottomleft(image: np.array, edge_color: int) -> np.array:
    """
    Raytrace in the downleft-direction, and determine what color is there.
    """
    skewed_image = image_skew(image, edge_color, SkewDirection.RIGHT)
    skewed_image = image_raytrace_probecolor_direction_bottom(skewed_image, edge_color)
    unskewed_image = image_unskew(skewed_image, SkewDirection.RIGHT)
    return unskewed_image

def image_raytrace_probecolor_direction_bottomright(image: np.array, edge_color: int) -> np.array:
    """
    Raytrace in the downright-direction, and determine what color is there.
    """
    skewed_image = image_skew(image, edge_color, SkewDirection.LEFT)
    skewed_image = image_raytrace_probecolor_direction_bottom(skewed_image, edge_color)
    unskewed_image = image_unskew(skewed_image, SkewDirection.LEFT)
    return unskewed_image

class ImageRaytraceProbeColorDirection(Enum):
    TOP = 'top'
    BOTTOM = 'bottom'
    LEFT = 'left'
    RIGHT = 'right'
    TOPLEFT = 'topleft'
    TOPRIGHT = 'topright'
    BOTTOMLEFT = 'bottomleft'
    BOTTOMRIGHT = 'bottomright'

def image_raytrace_probecolor_direction(image: np.array, edge_color: int, direction: ImageRaytraceProbeColorDirection) -> np.array:
    """
    Raytrace in the given direction, and determine what color is there.
    """
    if direction == ImageRaytraceProbeColorDirection.TOP:
        return image_raytrace_probecolor_direction_top(image, edge_color)
    elif direction == ImageRaytraceProbeColorDirection.BOTTOM:
        return image_raytrace_probecolor_direction_bottom(image, edge_color)
    elif direction == ImageRaytraceProbeColorDirection.LEFT:
        return image_raytrace_probecolor_direction_left(image, edge_color)
    elif direction == ImageRaytraceProbeColorDirection.RIGHT:
        return image_raytrace_probecolor_direction_right(image, edge_color)
    elif direction == ImageRaytraceProbeColorDirection.TOPLEFT:
        return image_raytrace_probecolor_direction_topleft(image, edge_color)
    elif direction == ImageRaytraceProbeColorDirection.TOPRIGHT:
        return image_raytrace_probecolor_direction_topright(image, edge_color)
    elif direction == ImageRaytraceProbeColorDirection.BOTTOMLEFT:
        return image_raytrace_probecolor_direction_bottomleft(image, edge_color)
    elif direction == ImageRaytraceProbeColorDirection.BOTTOMRIGHT:
        return image_raytrace_probecolor_direction_bottomright(image, edge_color)
    else:
        raise ValueError(f"Unknown direction: {direction}")
