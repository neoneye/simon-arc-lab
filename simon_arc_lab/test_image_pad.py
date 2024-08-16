import unittest
import numpy as np
from .image_pad import *

class TestImagePad(unittest.TestCase):
    def test_10000_image_pad_random_size1(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        actual = image_pad_random(image, seed=1, color=9, min_pad_count=1, max_pad_count=1)
        # Assert
        expected = np.array([
            [9, 9, 9, 9, 9],
            [9, 1, 2, 3, 9],
            [9, 4, 5, 6, 9],
            [9, 9, 9, 9, 9]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10001_image_pad_random_sizevary(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        actual = image_pad_random(image, seed=4, color=0, min_pad_count=1, max_pad_count=3)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 2, 3, 0, 0],
            [0, 0, 0, 4, 5, 6, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
