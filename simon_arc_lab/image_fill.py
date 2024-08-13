import numpy as np
from .pixel_connectivity import PixelConnectivity

class FloodFill:
    @staticmethod
    def _flood_fill_nearest4(image: np.array, x: int, y: int, from_color: int, to_color: int):
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
            return
        value = image[y, x]
        if value == to_color:
            return
        if value != from_color:
            return
        image[y, x] = to_color
        FloodFill._flood_fill_nearest4(image, x-1, y, from_color, to_color)
        FloodFill._flood_fill_nearest4(image, x+1, y, from_color, to_color)
        FloodFill._flood_fill_nearest4(image, x, y-1, from_color, to_color)
        FloodFill._flood_fill_nearest4(image, x, y+1, from_color, to_color)

    @staticmethod
    def _flood_fill_all8(image: np.array, x: int, y: int, from_color: int, to_color: int):
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
            return
        value = image[y, x]
        if value == to_color:
            return
        if value != from_color:
            return
        image[y, x] = to_color
        FloodFill._flood_fill_all8(image, x-1, y-1, from_color, to_color)
        FloodFill._flood_fill_all8(image, x, y-1, from_color, to_color)
        FloodFill._flood_fill_all8(image, x+1, y-1, from_color, to_color)
        FloodFill._flood_fill_all8(image, x-1, y, from_color, to_color)
        FloodFill._flood_fill_all8(image, x+1, y, from_color, to_color)
        FloodFill._flood_fill_all8(image, x-1, y+1, from_color, to_color)
        FloodFill._flood_fill_all8(image, x, y+1, from_color, to_color)
        FloodFill._flood_fill_all8(image, x+1, y+1, from_color, to_color)

    @staticmethod
    def _flood_fill_corner4(image: np.array, x: int, y: int, from_color: int, to_color: int):
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
            return
        value = image[y, x]
        if value == to_color:
            return
        if value != from_color:
            return
        image[y, x] = to_color
        FloodFill._flood_fill_corner4(image, x-1, y-1, from_color, to_color)
        FloodFill._flood_fill_corner4(image, x+1, y-1, from_color, to_color)
        FloodFill._flood_fill_corner4(image, x-1, y+1, from_color, to_color)
        FloodFill._flood_fill_corner4(image, x+1, y+1, from_color, to_color)

    @staticmethod
    def _mask_flood_fill_nearest4(mask: np.array, image: np.array, x: int, y: int, color: int):
        assert mask.shape == image.shape, "Both images must have the same size"
        if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
            return
        mask_value = mask[y, x]
        if mask_value > 0:
            return  # already visited
        value = image[y, x]
        if value != color:
            return
        mask[y, x] = 1  # flag as visited
        FloodFill._mask_flood_fill_nearest4(mask, image, x-1, y, color)
        FloodFill._mask_flood_fill_nearest4(mask, image, x+1, y, color)
        FloodFill._mask_flood_fill_nearest4(mask, image, x, y-1, color)
        FloodFill._mask_flood_fill_nearest4(mask, image, x, y+1, color)

    @staticmethod
    def _mask_flood_fill_all8(mask: np.array, image: np.array, x: int, y: int, color: int):
        assert mask.shape == image.shape, "Both images must have the same size"
        if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
            return
        mask_value = mask[y, x]
        if mask_value > 0:
            return  # already visited
        value = image[y, x]
        if value != color:
            return
        mask[y, x] = 1  # flag as visited
        FloodFill._mask_flood_fill_all8(mask, image, x-1, y-1, color)
        FloodFill._mask_flood_fill_all8(mask, image, x, y-1, color)
        FloodFill._mask_flood_fill_all8(mask, image, x+1, y-1, color)
        FloodFill._mask_flood_fill_all8(mask, image, x-1, y, color)
        FloodFill._mask_flood_fill_all8(mask, image, x+1, y, color)
        FloodFill._mask_flood_fill_all8(mask, image, x-1, y+1, color)
        FloodFill._mask_flood_fill_all8(mask, image, x, y+1, color)
        FloodFill._mask_flood_fill_all8(mask, image, x+1, y+1, color)

    @staticmethod
    def _mask_flood_fill_corner4(mask: np.array, image: np.array, x: int, y: int, color: int):
        assert mask.shape == image.shape, "Both images must have the same size"
        if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
            return
        mask_value = mask[y, x]
        if mask_value > 0:
            return  # already visited
        value = image[y, x]
        if value != color:
            return
        mask[y, x] = 1  # flag as visited
        FloodFill._mask_flood_fill_corner4(mask, image, x-1, y-1, color)
        FloodFill._mask_flood_fill_corner4(mask, image, x+1, y-1, color)
        FloodFill._mask_flood_fill_corner4(mask, image, x-1, y+1, color)
        FloodFill._mask_flood_fill_corner4(mask, image, x+1, y+1, color)

def image_flood_fill(image: np.array, x: int, y: int, from_color: int, to_color: int, connectivity: PixelConnectivity):
    """
    Replace color with another color.

    :param image: The image to modify in place.
    :param x: The x coordinate of the pixel to start the flood fill.
    :param y: The y coordinate of the pixel to start the flood fill.
    :param from_color: The color to replace.
    :param to_color: The color to replace with.
    :param connectivity: The pixel connectivity to use, how to visit neighbor pixels.
    """
    if connectivity == PixelConnectivity.NEAREST4:
        FloodFill._flood_fill_nearest4(image, x, y, from_color, to_color)
    elif connectivity == PixelConnectivity.ALL8:
        FloodFill._flood_fill_all8(image, x, y, from_color, to_color)
    elif connectivity == PixelConnectivity.CORNER4:
        FloodFill._flood_fill_corner4(image, x, y, from_color, to_color)
    else:
        raise ValueError("Invalid connectivity")

def image_mask_flood_fill(mask: np.array, image: np.array, x: int, y: int, color: int, connectivity: PixelConnectivity):
    """
    Build a mask of connected pixels that has the same color.

    :param mask: This mask is modified in place. The value of the mask is 1 where the flood fill occurred.
    :param image: The image to fill, however the image is not filled with any color.
    :param x: The x coordinate of the pixel to start the flood fill.
    :param y: The y coordinate of the pixel to start the flood fill.
    :param from_color: The color to replace.
    :param to_color: The color to replace with.
    :param connectivity: The pixel connectivity to use, how to visit neighbor pixels.
    """
    if connectivity == PixelConnectivity.NEAREST4:
        FloodFill._mask_flood_fill_nearest4(mask, image, x, y, color)
    elif connectivity == PixelConnectivity.ALL8:
        FloodFill._mask_flood_fill_all8(mask, image, x, y, color)
    elif connectivity == PixelConnectivity.CORNER4:
        FloodFill._mask_flood_fill_corner4(mask, image, x, y, color)
    else:
        raise ValueError("Invalid connectivity")
