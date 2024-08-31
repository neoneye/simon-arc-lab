import unittest
import numpy as np
from .image_skew import *

class TestImageSkew(unittest.TestCase):
    def test_10000_image_skew_left(self):
        # Arrange
        input = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        actual = image_skew_left(input, 255)
        # Assert
        expected = np.array([
            [255, 1, 2, 3],
            [4, 5, 6, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20000_image_unskew_left(self):
        # Arrange
        input = np.array([
            [255, 1, 2, 3],
            [4, 5, 6, 255]], dtype=np.uint8)
        # Act
        actual = image_unskew_left(input)
        # Assert
        expected = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_30000_image_skew_right(self):
        # Arrange
        input = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        actual = image_skew_right(input, 255)
        # Assert
        expected = np.array([
            [1, 2, 3, 255],
            [255, 4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_40000_image_unskew_right(self):
        # Arrange
        input = np.array([
            [1, 2, 3, 255],
            [255, 4, 5, 6]], dtype=np.uint8)
        # Act
        actual = image_unskew_right(input)
        # Assert
        expected = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_50000_image_skew_down(self):
        # Arrange
        input = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        # Act
        actual = image_skew_down(input, 255)
        # Assert
        expected = np.array([
            [1, 255],
            [2, 4],
            [3, 5],
            [255, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_60000_image_unskew_down(self):
        # Arrange
        input = np.array([
            [1, 255],
            [2, 4],
            [3, 5],
            [255, 6]], dtype=np.uint8)
        # Act
        actual = image_unskew_down(input)
        # Assert
        expected = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

