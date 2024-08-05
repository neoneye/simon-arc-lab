import numpy as np
from .pixel_connectivity import PixelConnectivity

class FloodFill:
    @staticmethod
    def _flood_fill4(image: np.array, x: int, y: int, from_color: int, to_color: int):
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
            return
        value = image[y, x]
        if value == to_color:
            return
        if value != from_color:
            return
        image[y, x] = to_color
        FloodFill._flood_fill4(image, x-1, y, from_color, to_color)
        FloodFill._flood_fill4(image, x+1, y, from_color, to_color)
        FloodFill._flood_fill4(image, x, y-1, from_color, to_color)
        FloodFill._flood_fill4(image, x, y+1, from_color, to_color)

    @staticmethod
    def _flood_fill8(image: np.array, x: int, y: int, from_color: int, to_color: int):
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
            return
        value = image[y, x]
        if value == to_color:
            return
        if value != from_color:
            return
        image[y, x] = to_color
        FloodFill._flood_fill8(image, x-1, y-1, from_color, to_color)
        FloodFill._flood_fill8(image, x, y-1, from_color, to_color)
        FloodFill._flood_fill8(image, x+1, y-1, from_color, to_color)
        FloodFill._flood_fill8(image, x-1, y, from_color, to_color)
        FloodFill._flood_fill8(image, x+1, y, from_color, to_color)
        FloodFill._flood_fill8(image, x-1, y+1, from_color, to_color)
        FloodFill._flood_fill8(image, x, y+1, from_color, to_color)
        FloodFill._flood_fill8(image, x+1, y+1, from_color, to_color)

    @staticmethod
    def _mask_flood_fill4(mask: np.array, image: np.array, x: int, y: int, color: int):
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
        FloodFill._mask_flood_fill4(mask, image, x-1, y, color)
        FloodFill._mask_flood_fill4(mask, image, x+1, y, color)
        FloodFill._mask_flood_fill4(mask, image, x, y-1, color)
        FloodFill._mask_flood_fill4(mask, image, x, y+1, color)

    @staticmethod
    def _mask_flood_fill8(mask: np.array, image: np.array, x: int, y: int, color: int):
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
        FloodFill._mask_flood_fill8(mask, image, x-1, y-1, color)
        FloodFill._mask_flood_fill8(mask, image, x, y-1, color)
        FloodFill._mask_flood_fill8(mask, image, x+1, y-1, color)
        FloodFill._mask_flood_fill8(mask, image, x-1, y, color)
        FloodFill._mask_flood_fill8(mask, image, x+1, y, color)
        FloodFill._mask_flood_fill8(mask, image, x-1, y+1, color)
        FloodFill._mask_flood_fill8(mask, image, x, y+1, color)
        FloodFill._mask_flood_fill8(mask, image, x+1, y+1, color)

def image_flood_fill(image: np.array, x: int, y: int, from_color: int, to_color: int, connectivity: PixelConnectivity):
    """
    Replace color with another color.

    Visit 4 or 8 neighbors around a pixel.
    """
    if connectivity == PixelConnectivity.CONNECTIVITY4:
        FloodFill._flood_fill4(image, x, y, from_color, to_color)
    elif connectivity == PixelConnectivity.CONNECTIVITY8:
        FloodFill._flood_fill8(image, x, y, from_color, to_color)
    else:
        raise ValueError("Invalid connectivity")

def image_mask_flood_fill(mask: np.array, image: np.array, x: int, y: int, color: int, connectivity: PixelConnectivity):
    """
    Build a mask of connected pixels that has the same color.

    Visit 4 or 8 neighbors around a pixel.
    """
    if connectivity == PixelConnectivity.CONNECTIVITY4:
        FloodFill._mask_flood_fill4(mask, image, x, y, color)
    elif connectivity == PixelConnectivity.CONNECTIVITY8:
        FloodFill._mask_flood_fill8(mask, image, x, y, color)
    else:
        raise ValueError("Invalid connectivity")
