import unittest
import numpy as np
from .image_outline import *

class TestImageOutline(unittest.TestCase):
    def test_10000_image_outline_all8_the_same_color(self):
        # Arrange
        image = np.array([
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9]], dtype=np.uint8)
        # Act
        actual = image_outline_all8(image)
        # Assert
        expected = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10001_image_outline_all8_staircase(self):
        # Arrange
        image = np.array([
            [5, 5,     9, 9, 9, 9, 9],
            [5, 5, 5,     9, 9, 9, 9],
            [5, 5, 5, 5,     9, 9, 9],
            [5, 5, 5, 5, 5,     9, 9]], dtype=np.uint8)
        # Act
        actual = image_outline_all8(image)
        # Assert
        expected = np.array([
            [1, 1,     1, 1, 1, 1, 1],
            [1, 1, 1,     1, 1, 0, 1],
            [1, 0, 1, 1,     1, 1, 1],
            [1, 1, 1, 1, 1,     1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10002_image_outline_all8_filled_rects(self):
        # Arrange
        image = np.array([
            [9, 9, 9, 7, 9, 9, 9, 9, 9],
            [9, 9, 9, 7, 9, 9, 9, 9, 9],
            [9, 9, 9, 7, 9, 9, 9, 9, 9],
            [9, 9, 9, 7, 7, 7, 7, 7, 7],
            [9, 9, 9, 7, 7, 7, 7, 7, 7],
            [9, 9, 9, 7, 7, 7, 7, 7, 7],
            [9, 9, 9, 7, 7, 7, 7, 7, 7]], dtype=np.uint8)
        # Act
        actual = image_outline_all8(image)
        # Assert
        expected = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

