import unittest
import numpy as np
from .image_erosion_multicolor import *
from .pixel_connectivity import PixelConnectivity

class TestImageErodeMultiColor(unittest.TestCase):
    def test_10000_image_erosion_multicolor_all8(self):
        # Arrange
        image = np.array([
            [9, 9, 9, 2, 2, 2],
            [9, 9, 9, 2, 2, 2],
            [9, 9, 9, 2, 2, 2],
            [7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7]], dtype=np.uint8)
        # Act
        actual = image_erosion_multicolor(image, PixelConnectivity.ALL8)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_11000_image_erosion_multicolor_lr2(self):
        # Arrange
        image = np.array([
            [9, 9, 9, 2, 2, 2],
            [9, 9, 9, 2, 2, 2],
            [9, 9, 9, 2, 2, 2],
            [7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7]], dtype=np.uint8)
        # Act
        actual = image_erosion_multicolor(image, PixelConnectivity.LR2)
        # Assert
        expected = np.array([
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
