import unittest
import numpy as np
from .image_fractal import *

class TestImageFractal(unittest.TestCase):
    def test_10000_image_fractal_from_mask(self):
        # Arrange
        image = np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 0]], dtype=np.uint8)
        # Act
        actual = image_fractal_from_mask(image)
        # Assert
        expected = np.array([
            [1, 1, 0,  1, 1, 0,  0, 0, 0],
            [1, 0, 1,  1, 0, 1,  0, 0, 0],
            [0, 1, 0,  0, 1, 0,  0, 0, 0],

            [1, 1, 0,  0, 0, 0,  1, 1, 0],
            [1, 0, 1,  0, 0, 0,  1, 0, 1],
            [0, 1, 0,  0, 0, 0,  0, 1, 0],

            [0, 0, 0,  1, 1, 0,  0, 0, 0],
            [0, 0, 0,  1, 0, 1,  0, 0, 0],
            [0, 0, 0,  0, 1, 0,  0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10001_image_fractal_from_mask(self):
        # Arrange
        image = np.array([
            [1, 0, 0],
            [0, 0, 1]], dtype=np.uint8)
        # Act
        actual = image_fractal_from_mask(image)
        # Assert
        expected = np.array([
            [1, 0, 0,  0, 0, 0,  0, 0, 0],
            [0, 0, 1,  0, 0, 0,  0, 0, 0],

            [0, 0, 0,  0, 0, 0,  1, 0, 0],
            [0, 0, 0,  0, 0, 0,  0, 0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
