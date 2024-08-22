import unittest
import numpy as np
from .image_raytrace import *

class TestImageRaytrace(unittest.TestCase):
    def test_10000_image_raytrace_probe_color_direction_up(self):
        # Arrange
        image = np.array([
            [0, 0, 2, 0],
            [0, 1, 2, 3],
            [0, 1, 2, 0],
            [0, 4, 5, 6],
            [0, 0, 5, 0]], dtype=np.uint8)
        # Act
        actual = image_raytrace_probe_color_direction_up(image, 10)
        # Assert
        expected = np.array([
            [10, 10, 10, 10],
            [10,  0, 10,  0],
            [10,  0, 10,  3],
            [10,  1,  2,  0],
            [10,  4,  2,  6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_11000_image_raytrace_probe_color_direction_down(self):
        # Arrange
        image = np.array([
            [0, 0, 2, 0],
            [0, 1, 2, 3],
            [0, 1, 2, 0],
            [0, 4, 5, 6],
            [0, 0, 5, 0]], dtype=np.uint8)
        # Act
        actual = image_raytrace_probe_color_direction_down(image, 10)
        # Assert
        expected = np.array([
            [10,  1,  5,  3],
            [10,  4,  5,  0],
            [10,  4,  5,  6],
            [10,  0, 10,  0],
            [10, 10, 10, 10]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_12000_image_raytrace_probe_color_direction_left(self):
        # Arrange
        image = np.array([
            [0, 3, 0, 6, 0],
            [2, 2, 2, 5, 5],
            [0, 1, 1, 4, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_raytrace_probe_color_direction_left(image, 10)
        # Assert
        expected = np.array([
            [10,  0,  3,  0,  6],
            [10, 10, 10,  2,  2],
            [10,  0,  0,  1,  4],
            [10, 10, 10, 10, 10]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_13000_image_raytrace_probe_color_direction_right(self):
        # Arrange
        image = np.array([
            [0, 3, 0, 6, 0],
            [2, 2, 2, 5, 5],
            [0, 1, 1, 4, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_raytrace_probe_color_direction_right(image, 10)
        # Assert
        expected = np.array([
            [ 3,  0,  6,  0, 10],
            [ 5,  5,  5, 10, 10],
            [ 1,  4,  4,  0, 10],
            [10, 10, 10, 10, 10]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
