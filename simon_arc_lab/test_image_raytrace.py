import unittest
import numpy as np
from .image_raytrace import *

class TestImageRaytrace(unittest.TestCase):
    def test_10000_image_raytrace_probe_color_direction_top(self):
        # Arrange
        image = np.array([
            [0, 0, 2, 0],
            [0, 1, 2, 3],
            [0, 1, 2, 0],
            [0, 4, 5, 6],
            [0, 0, 5, 0]], dtype=np.uint8)
        # Act
        actual = image_raytrace_probe_color_direction_top(image, 10)
        # Assert
        expected = np.array([
            [10, 10, 10, 10],
            [10,  0, 10,  0],
            [10,  0, 10,  3],
            [10,  1,  2,  0],
            [10,  4,  2,  6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_11000_image_raytrace_probe_color_direction_bottom(self):
        # Arrange
        image = np.array([
            [0, 0, 2, 0],
            [0, 1, 2, 3],
            [0, 1, 2, 0],
            [0, 4, 5, 6],
            [0, 0, 5, 0]], dtype=np.uint8)
        # Act
        actual = image_raytrace_probe_color_direction_bottom(image, 10)
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

    def test_14000_image_raytrace_probe_color_direction_topleft(self):
        # Arrange
        image = np.array([
            [1, 0, 0, 2],
            [0, 1, 2, 3],
            [0, 1, 1, 0],
            [0, 4, 5, 6],
            [4, 0, 5, 3]], dtype=np.uint8)
        # Act
        actual = image_raytrace_probe_color_direction_topleft(image, 10)
        # Assert
        expected = np.array([
            [10, 10, 10, 10],
            [10, 10,  0,  0],
            [10,  0, 10,  2],
            [10,  0,  1,  1],
            [10, 10,  4,  5]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_15000_image_raytrace_probe_color_direction_topright(self):
        # Arrange
        image = np.array([
            [1, 0, 0, 2],
            [0, 1, 2, 3],
            [0, 1, 1, 0],
            [0, 4, 5, 6],
            [4, 0, 5, 3]], dtype=np.uint8)
        # Act
        actual = image_raytrace_probe_color_direction_topright(image, 10)
        # Assert
        expected = np.array([
            [10, 10, 10, 10],
            [10,  0, 10, 10],
            [ 1,  2,  3, 10],
            [ 1,  1,  0, 10],
            [ 1,  5,  6, 10]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_16000_image_raytrace_probe_color_direction_bottomleft(self):
        # Arrange
        image = np.array([
            [1, 0, 0, 2],
            [0, 1, 2, 3],
            [0, 1, 1, 0],
            [0, 4, 5, 6],
            [4, 0, 5, 3]], dtype=np.uint8)
        # Act
        actual = image_raytrace_probe_color_direction_bottomleft(image, 10)
        # Assert
        expected = np.array([
            [10, 10,  1,  1],
            [10,  0,  1,  1],
            [10,  0,  4,  5],
            [10, 10,  0,  5],
            [10, 10, 10, 10]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_17000_image_raytrace_probe_color_direction_bottomright(self):
        # Arrange
        image = np.array([
            [1, 0, 0, 2],
            [0, 1, 2, 3],
            [0, 1, 1, 0],
            [0, 4, 5, 6],
            [4, 0, 5, 3]], dtype=np.uint8)
        # Act
        actual = image_raytrace_probe_color_direction_bottomright(image, 10)
        # Assert
        expected = np.array([
            [ 6,  2,  3, 10],
            [ 1,  6,  0, 10],
            [ 4,  5,  6, 10],
            [10,  5,  3, 10],
            [10, 10, 10, 10]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

