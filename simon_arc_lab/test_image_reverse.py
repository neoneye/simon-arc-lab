import unittest
import numpy as np
from .image_reverse import *

class TestImageReverse(unittest.TestCase):
    def test_10000_image_reverse_leftright(self):
        # Arrange
        input = np.array([
            [1, 2, 3, 4, 5],
            [0, 0, 1, 2, 3],
            [1, 2, 3, 0, 0],
            [1, 2, 0, 1, 2],
            [1, 1, 2, 3, 3]], dtype=np.uint8)
        # Act
        actual = image_reverse(input, 0, ReverseDirection.LEFTRIGHT)
        # Assert
        expected = np.array([
            [5, 4, 3, 2, 1],
            [0, 0, 3, 2, 1],
            [3, 2, 1, 0, 0],
            [2, 1, 0, 2, 1],
            [3, 3, 2, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10001_image_reverse_topbottom(self):
        # Arrange
        input = np.array([
            [1, 3, 3, 5, 7],
            [1, 1, 3, 0, 8],
            [1, 3, 2, 5, 9],
            [1, 0, 2, 0, 9],
            [2, 0, 0, 5, 0]], dtype=np.uint8)
        # Act
        actual = image_reverse(input, 0, ReverseDirection.TOPBOTTOM)
        # Assert
        expected = np.array([
            [2, 3, 2, 5, 9],
            [1, 1, 2, 0, 9],
            [1, 3, 3, 5, 8],
            [1, 0, 3, 0, 7],
            [1, 0, 0, 5, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

