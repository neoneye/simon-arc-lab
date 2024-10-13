import unittest
import numpy as np
from .image_tile_template import *
from .image_util import image_replace_colors, image_flipx

class TestImageTileTemplate(unittest.TestCase):
    def test_10000_image_tile_template_without_callback(self):
        # Arrange
        tile_layout = np.array([
            [3, 4, 5]
        ], dtype=np.uint8)
        base_tile = np.array([
            [0, 1], 
            [1, 1], 
            [0, 1]], dtype=np.uint8)
        # Act
        actual = image_tile_template(base_tile, tile_layout)
        # Assert
        expected = np.array([
            [3, 1, 4, 1, 5, 1],
            [1, 1, 1, 1, 1, 1],
            [3, 1, 4, 1, 5, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_11000_image_tile_template_with_callback(self):
        # Arrange
        tile_layout = np.array([
            [3, 4, 5]
        ], dtype=np.uint8)
        base_tile = np.array([
            [0, 1], 
            [1, 1], 
            [0, 1]], dtype=np.uint8)
        # Act
        def callback(tile, layout, x, y):
            color_map = { 1: layout[y, x] }
            return image_replace_colors(tile, color_map)
        actual = image_tile_template(base_tile, tile_layout, callback)
        # Assert
        expected = np.array([
            [0, 3, 0, 4, 0, 5],
            [3, 3, 4, 4, 5, 5],
            [0, 3, 0, 4, 0, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_11001_image_tile_template_with_callback(self):
        # Arrange
        tile_layout = np.array([
            [3, 4, 5, 6]
        ], dtype=np.uint8)
        base_tile = np.array([
            [0, 1], 
            [1, 1], 
            [0, 1]], dtype=np.uint8)
        # Act
        def callback(tile, layout, x, y):
            color_map = { 1: layout[y, x] }
            image = image_replace_colors(tile, color_map)
            if x & 1 == 1:
                image = image_flipx(image)
            return image
        actual = image_tile_template(base_tile, tile_layout, callback)
        # Assert
        expected = np.array([
            [0, 3, 4, 0, 0, 5, 6, 0],
            [3, 3, 4, 4, 5, 5, 6, 6],
            [0, 3, 4, 0, 0, 5, 6, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
