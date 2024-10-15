import numpy as np
from .image_util import image_replace_colors

def image_tile_template(base_tile: np.array, tile_layout: np.array, callback=None) -> np.array:
    """
    Create a repetitive pattern with a customizable tile transformation.

    :param base_tile: The foundational tile upon which the pattern is built.
    :param tile_layout: This defines the spatial layout of tiles.
    :param callback: A function that transforms the tile based on x and y coordinates.
    :return: The tiled pattern image.
    """
    if callback is None:
        def callback(tile, layout, x, y):
            color_map = {0: layout[y, x]}
            return image_replace_colors(tile, color_map)

    tile_height, tile_width = base_tile.shape
    layout_height, layout_width = tile_layout.shape
    result = np.zeros((tile_height * layout_height, tile_width * layout_width), dtype=np.uint8)
    for y in range(layout_height):
        for x in range(layout_width):
            tile_transformed = callback(base_tile, tile_layout, x, y)
            result[y * tile_height:(y + 1) * tile_height, x * tile_width:(x + 1) * tile_width] = tile_transformed
    return result
