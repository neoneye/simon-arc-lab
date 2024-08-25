import unittest
import numpy as np
from .image_pattern import *

class TestImagePattern(unittest.TestCase):
    def test_10000_checkerboard_colors2_size1_offset00(self):
        # Act
        actual = image_pattern_checkerboard(5, 4, 1, 0, 0, [5, 6])
        # Assert
        expected = np.array([
            [5, 6, 5, 6, 5],
            [6, 5, 6, 5, 6],
            [5, 6, 5, 6, 5],
            [6, 5, 6, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10001_checkerboard_colors2_size2_offset00(self):
        # Act
        actual = image_pattern_checkerboard(5, 4, 2, 0, 0, [0, 1])
        # Assert
        expected = np.array([
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10002_checkerboard_colors2_size2_offset10(self):
        # Act
        actual = image_pattern_checkerboard(5, 4, 2, 1, 0, [0, 1])
        # Assert
        expected = np.array([
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 1],
            [1, 0, 0, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10003_checkerboard_colors2_size2_offset01(self):
        # Act
        actual = image_pattern_checkerboard(5, 4, 2, 0, 1, [0, 1])
        # Assert
        expected = np.array([
            [0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10004_checkerboard_colors2_size3_offset00(self):
        # Act
        actual = image_pattern_checkerboard(5, 4, 3, 0, 0, [0, 1])
        # Assert
        expected = np.array([
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10005_checkerboard_colors3_size2_offset00(self):
        # Act
        actual = image_pattern_checkerboard(7, 5, 2, 0, 0, [0, 1, 2])
        # Assert
        expected = np.array([
            [0, 0, 1, 1, 2, 2, 0],
            [0, 0, 1, 1, 2, 2, 0],
            [1, 1, 2, 2, 0, 0, 1],
            [1, 1, 2, 2, 0, 0, 1],
            [2, 2, 0, 0, 1, 1, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

