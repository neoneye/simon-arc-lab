import unittest
import numpy as np
from .image_rect import *
from .rectangle import *

class TestImageRect(unittest.TestCase):
    def test_10000_image_rect_all_of_rect_is_inside(self):
        # Arrange
        image = np.array([
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        # Act
        actual = image_rect(image, Rectangle(1, 2, 3, 4), 2)
        # Assert
        expected = np.array([
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 9, 9, 9, 9],
            [9, 2, 2, 2, 9, 9, 9, 9],
            [9, 2, 2, 2, 9, 9, 9, 9],
            [9, 2, 2, 2, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10001_image_rect_with_big_rect(self):
        # Arrange
        image = np.array([
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9]], dtype=np.uint8)
        # Act
        actual = image_rect(image, Rectangle(-10, -10, 30, 30), 2)
        # Assert
        expected = np.array([
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20000_image_rect_hollow_all_of_rect_is_inside_size1(self):
        # Arrange
        image = np.array([
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        # Act
        actual = image_rect_hollow(image, Rectangle(1, 2, 3, 4), 2, 1)
        # Assert
        expected = np.array([
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 9, 9, 9, 9],
            [9, 2, 9, 2, 9, 9, 9, 9],
            [9, 2, 9, 2, 9, 9, 9, 9],
            [9, 2, 2, 2, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20001_image_rect_hollow_all_of_rect_is_inside_size2(self):
        # Arrange
        image = np.array([
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9]], dtype=np.uint8)
        # Act
        actual = image_rect_hollow(image, Rectangle(1, 2, 5, 6), 2, 2)
        # Assert
        expected = np.array([
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 9, 9],
            [9, 2, 2, 2, 2, 2, 9, 9],
            [9, 2, 2, 9, 2, 2, 9, 9],
            [9, 2, 2, 9, 2, 2, 9, 9],
            [9, 2, 2, 2, 2, 2, 9, 9]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

