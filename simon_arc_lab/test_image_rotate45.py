import unittest
import numpy as np
from .image_rotate45 import *

class TestImageRotate45(unittest.TestCase):
    def test_10000_rotate_tiny_images(self):
        # Empty image
        original = np.array([], dtype=np.uint8).reshape(0, 0)
        actual = image_rotate_cw_45(original, 0)
        np.testing.assert_array_equal(actual, original)

        # 1x1 image
        original = np.array([[9]], dtype=np.uint8)
        actual = image_rotate_cw_45(original, 0)
        np.testing.assert_array_equal(actual, original)

    def test_10001_rotate_ccw_square(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype=np.uint8)

        # Act
        actual = image_rotate_ccw_45(image, 0)

        # Assert
        expected = np.array([
            [0, 0, 3, 0, 0],
            [0, 2, 0, 6, 0],
            [1, 0, 5, 0, 9],
            [0, 4, 0, 8, 0],
            [0, 0, 7, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10002_rotate_ccw_landscape_onerow(self):
        # Arrange
        image = np.array([[1, 2, 3]], dtype=np.uint8)

        # Act
        actual = image_rotate_ccw_45(image, 0)

        # Assert
        expected = np.array([
            [0, 0, 3],
            [0, 2, 0],
            [1, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10003_rotate_ccw_landscape_tworows(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)

        # Act
        actual = image_rotate_ccw_45(image, 0)

        # Assert
        expected = np.array([
            [0, 0, 3, 0],
            [0, 2, 0, 6],
            [1, 0, 5, 0],
            [0, 4, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10004_rotate_ccw_portrait_onecolumn(self):
        # Arrange
        image = np.array([
            [1],
            [2],
            [3]], dtype=np.uint8)

        # Act
        actual = image_rotate_ccw_45(image, 0)

        # Assert
        expected = np.array([
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10005_rotate_ccw_portrait_twocolumns(self):
        # Arrange
        image = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)

        # Act
        actual = image_rotate_ccw_45(image, 0)

        # Assert
        expected = np.array([
            [0, 4, 0, 0],
            [1, 0, 5, 0],
            [0, 2, 0, 6],
            [0, 0, 3, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20000_rotate_cw(self):
        # Arrange
        image = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)

        # Act
        actual = image_rotate_cw_45(image, 0)

        # Assert
        expected = np.array([
            [0, 0, 1, 0],
            [0, 2, 0, 4],
            [3, 0, 5, 0],
            [0, 6, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
