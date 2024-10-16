import numpy as np
from .pixel_connectivity import PixelConnectivity

class FloodFill:
    @staticmethod
    def _flood_fill_nearest4(image: np.array, x: int, y: int, from_color: int, to_color: int):
        if from_color == to_color:
            return  # No action needed if the colors are the same

        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            # Check if the coordinates are within the image bounds
            if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                continue
            value = image[y, x]
            if value == to_color:
                continue # Already filled with the target color
            if value != from_color:
                continue # Not the color we're looking to replace
            image[y, x] = to_color  # Fill the pixel with the target color
            # Add the 4-connected neighbors to the stack
            neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            stack.extend(neighbors)

    @staticmethod
    def _flood_fill_all8(image: np.array, x: int, y: int, from_color: int, to_color: int):
        if from_color == to_color:
            return  # No action needed if the colors are the same

        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            # Check if the coordinates are within the image bounds
            if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                continue
            value = image[y, x]
            if value == to_color:
                continue # Already filled with the target color
            if value != from_color:
                continue # Not the color we're looking to replace
            image[y, x] = to_color  # Fill the pixel with the target color
            # Add all 8-connected neighbors to the stack
            neighbors = [
                (x-1, y-1), (x, y-1), (x+1, y-1),
                (x-1, y),             (x+1, y),
                (x-1, y+1), (x, y+1), (x+1, y+1)
            ]
            stack.extend(neighbors)

    @staticmethod
    def _flood_fill_corner4(image: np.array, x: int, y: int, from_color: int, to_color: int):
        if from_color == to_color:
            return  # No action needed if the colors are the same

        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            # Check if the coordinates are within the image bounds
            if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                continue
            value = image[y, x]
            if value == to_color:
                continue # Already filled with the target color
            if value != from_color:
                continue # Not the color we're looking to replace
            image[y, x] = to_color  # Fill the pixel with the target color
            # Add all 4 corners to the stack
            neighbors = [(x-1, y-1), (x+1, y-1), (x-1, y+1), (x+1, y+1)]
            stack.extend(neighbors)

    @staticmethod
    def _flood_fill_lr2(image: np.array, x: int, y: int, from_color: int, to_color: int):
        if from_color == to_color:
            return  # No action needed if the colors are the same

        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            # Check if the coordinates are within the image bounds
            if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                continue
            value = image[y, x]
            if value == to_color:
                continue # Already filled with the target color
            if value != from_color:
                continue # Not the color we're looking to replace
            image[y, x] = to_color  # Fill the pixel with the target color
            # Add two neighbors to the stack
            neighbors = [(x-1, y), (x+1, y)]
            stack.extend(neighbors)

    @staticmethod
    def _flood_fill_tb2(image: np.array, x: int, y: int, from_color: int, to_color: int):
        if from_color == to_color:
            return  # No action needed if the colors are the same

        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            # Check if the coordinates are within the image bounds
            if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                continue
            value = image[y, x]
            if value == to_color:
                continue # Already filled with the target color
            if value != from_color:
                continue # Not the color we're looking to replace
            image[y, x] = to_color  # Fill the pixel with the target color
            # Add two neighbors to the stack
            neighbors = [(x, y-1), (x, y+1)]
            stack.extend(neighbors)

    @staticmethod
    def _flood_fill_tlbr2(image: np.array, x: int, y: int, from_color: int, to_color: int):
        if from_color == to_color:
            return  # No action needed if the colors are the same

        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            # Check if the coordinates are within the image bounds
            if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                continue
            value = image[y, x]
            if value == to_color:
                continue # Already filled with the target color
            if value != from_color:
                continue # Not the color we're looking to replace
            image[y, x] = to_color  # Fill the pixel with the target color
            # Add two neighbors to the stack
            neighbors = [(x-1, y-1), (x+1, y+1)]
            stack.extend(neighbors)

    @staticmethod
    def _flood_fill_trbl2(image: np.array, x: int, y: int, from_color: int, to_color: int):
        if from_color == to_color:
            return  # No action needed if the colors are the same

        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            # Check if the coordinates are within the image bounds
            if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
                continue
            value = image[y, x]
            if value == to_color:
                continue # Already filled with the target color
            if value != from_color:
                continue # Not the color we're looking to replace
            image[y, x] = to_color  # Fill the pixel with the target color
            # Add two neighbors to the stack
            neighbors = [(x+1, y-1), (x-1, y+1)]
            stack.extend(neighbors)

    @staticmethod
    def _mask_flood_fill_nearest4(mask: np.array, image: np.array, x: int, y: int, color: int):
        assert mask.shape == image.shape, "Both images must have the same size"
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
                continue
            if mask[y, x] > 0:
                continue # already visited
            if image[y, x] != color:
                continue
            mask[y, x] = 1 # flag as visited
            # Add the 4-connected neighbors to the stack
            neighbors = [(x, y-1), (x-1, y), (x+1, y), (x, y+1)]
            stack.extend(neighbors)

    @staticmethod
    def _mask_flood_fill_all8(mask: np.array, image: np.array, x: int, y: int, color: int):
        assert mask.shape == image.shape, "Both images must have the same size"
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
                continue
            if mask[y, x] > 0:
                continue # already visited
            if image[y, x] != color:
                continue
            mask[y, x] = 1 # flag as visited
            # Add all 8-connected neighbors to the stack
            neighbors = [
                (x-1, y-1), (x, y-1), (x+1, y-1),
                (x-1, y),             (x+1, y),
                (x-1, y+1), (x, y+1), (x+1, y+1)
            ]
            stack.extend(neighbors)

    @staticmethod
    def _mask_flood_fill_corner4(mask: np.array, image: np.array, x: int, y: int, color: int):
        assert mask.shape == image.shape, "Both images must have the same size"
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
                continue
            if mask[y, x] > 0:
                continue  # already visited
            if image[y, x] != color:
                continue
            mask[y, x] = 1 # flag as visited
            # Add corner neighbors to the stack
            neighbors = [(x-1, y-1), (x+1, y-1), (x-1, y+1), (x+1, y+1)]
            stack.extend(neighbors)

    @staticmethod
    def _mask_flood_fill_lr2(mask: np.array, image: np.array, x: int, y: int, color: int):
        assert mask.shape == image.shape, "Both images must have the same size"
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
                continue
            if mask[y, x] > 0:
                continue  # already visited
            if image[y, x] != color:
                continue
            mask[y, x] = 1 # flag as visited
            # Add two neighbors to the stack
            neighbors = [(x-1, y), (x+1, y)]
            stack.extend(neighbors)

    @staticmethod
    def _mask_flood_fill_tb2(mask: np.array, image: np.array, x: int, y: int, color: int):
        assert mask.shape == image.shape, "Both images must have the same size"
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
                continue
            if mask[y, x] > 0:
                continue  # already visited
            if image[y, x] != color:
                continue
            mask[y, x] = 1 # flag as visited
            # Add two neighbors to the stack
            neighbors = [(x, y-1), (x, y+1)]
            stack.extend(neighbors)

    @staticmethod
    def _mask_flood_fill_tlbr2(mask: np.array, image: np.array, x: int, y: int, color: int):
        assert mask.shape == image.shape, "Both images must have the same size"
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
                continue
            if mask[y, x] > 0:
                continue  # already visited
            if image[y, x] != color:
                continue
            mask[y, x] = 1 # flag as visited
            # Add two neighbors to the stack
            neighbors = [(x-1, y-1), (x+1, y+1)]
            stack.extend(neighbors)

    @staticmethod
    def _mask_flood_fill_trbl2(mask: np.array, image: np.array, x: int, y: int, color: int):
        assert mask.shape == image.shape, "Both images must have the same size"
        stack = [(x, y)]
        while stack:
            x, y = stack.pop()
            if x < 0 or y < 0 or x >= mask.shape[1] or y >= mask.shape[0]:
                continue
            if mask[y, x] > 0:
                continue  # already visited
            if image[y, x] != color:
                continue
            mask[y, x] = 1 # flag as visited
            # Add two neighbors to the stack
            neighbors = [(x+1, y-1), (x-1, y+1)]
            stack.extend(neighbors)

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
    elif connectivity == PixelConnectivity.LR2:
        FloodFill._flood_fill_lr2(image, x, y, from_color, to_color)
    elif connectivity == PixelConnectivity.TB2:
        FloodFill._flood_fill_tb2(image, x, y, from_color, to_color)
    elif connectivity == PixelConnectivity.TLBR2:
        FloodFill._flood_fill_tlbr2(image, x, y, from_color, to_color)
    elif connectivity == PixelConnectivity.TRBL2:
        FloodFill._flood_fill_trbl2(image, x, y, from_color, to_color)
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
    elif connectivity == PixelConnectivity.LR2:
        FloodFill._mask_flood_fill_lr2(mask, image, x, y, color)
    elif connectivity == PixelConnectivity.TB2:
        FloodFill._mask_flood_fill_tb2(mask, image, x, y, color)
    elif connectivity == PixelConnectivity.TLBR2:
        FloodFill._mask_flood_fill_tlbr2(mask, image, x, y, color)
    elif connectivity == PixelConnectivity.TRBL2:
        FloodFill._mask_flood_fill_trbl2(mask, image, x, y, color)
    else:
        raise ValueError("Invalid connectivity")
