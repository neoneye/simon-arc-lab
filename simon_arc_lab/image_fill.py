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

def image_fill_inplace(image: np.array, x: int, y: int, from_color: int, to_color: int, connectivity: PixelConnectivity):
    """
    Flood fill
    """
    if connectivity == PixelConnectivity.CONNECTIVITY4:
        FloodFill._flood_fill4(image, x, y, from_color, to_color)
    elif connectivity == PixelConnectivity.CONNECTIVITY8:
        FloodFill._flood_fill8(image, x, y, from_color, to_color)
    else:
        raise ValueError("Invalid connectivity")
