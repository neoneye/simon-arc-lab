import unittest
import numpy as np
from .image_trim import *
from .rectangle import *

class TestImageTrim(unittest.TestCase):
    def test_10000_find_bounding_box_multiple_ignore_colors(self):
        # Arrange
        image = np.array([
            [8, 8, 8, 8],
            [8, 8, 8, 8],
            [8, 1, 2, 8],
            [8, 3, 4, 8],
            [9, 5, 6, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9]], dtype=np.uint8)
        # Act
        actual = find_bounding_box_multiple_ignore_colors(image, [7, 8, 9])
        # Assert
        expected = Rectangle(1, 2, 2, 3)
        self.assertEqual(actual, expected)

    def test_20000_find_bounding_box_ignoring_color(self):
        # Arrange
        image = np.array([
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 1, 2, 3, 9, 9, 9],
            [9, 9, 4, 5, 6, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        # Act
        actual = find_bounding_box_ignoring_color(image, 9)
        # Assert
        expected = Rectangle(2, 1, 3, 2)
        self.assertEqual(actual, expected)

    def test_20001_find_bounding_box_ignoring_color(self):
        # Arrange
        image = np.array([
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 1, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 1, 9],
            [9, 9, 9, 1, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        # Act
        actual = find_bounding_box_ignoring_color(image, 9)
        # Assert
        expected = Rectangle(1, 2, 6, 4)
        self.assertEqual(actual, expected)

    def test_20002_find_bounding_box_ignoring_color_empty(self):
        # Arrange
        image = np.array([
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9]], dtype=np.uint8)
        # Act
        actual = find_bounding_box_ignoring_color(image, 9)
        # Assert
        expected = Rectangle.empty()
        self.assertEqual(actual, expected)

    def test_20003_find_bounding_box_ignoring_color_empty(self):
        # Arrange
        image = np.array([], dtype=np.uint8)
        # Act
        actual = find_bounding_box_ignoring_color(image, 9)
        # Assert
        expected = Rectangle.empty()
        self.assertEqual(actual, expected)
