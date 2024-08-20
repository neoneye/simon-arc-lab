import unittest
import numpy as np
from .image_trim import *
from .rectangle import *

class TestImageTrim(unittest.TestCase):
    def test_10000_outer_bounding_box1(self):
        # Arrange
        image = np.array([
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 1, 2, 3, 9, 9, 9],
            [9, 9, 4, 5, 6, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        # Act
        actual = outer_bounding_box_after_trim_with_color(image, 9)
        # Assert
        expected = Rectangle(2, 1, 3, 2)
        self.assertEqual(actual, expected)

    def test_10001_outer_bounding_box2(self):
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
        actual = outer_bounding_box_after_trim_with_color(image, 9)
        # Assert
        expected = Rectangle(1, 2, 6, 4)
        self.assertEqual(actual, expected)

    def test_10002_outer_bounding_box_empty(self):
        # Arrange
        image = np.array([
            [9, 9, 9, 9],
            [9, 9, 9, 9],
            [9, 9, 9, 9]], dtype=np.uint8)
        # Act
        actual = outer_bounding_box_after_trim_with_color(image, 9)
        # Assert
        expected = Rectangle.empty()
        self.assertEqual(actual, expected)
