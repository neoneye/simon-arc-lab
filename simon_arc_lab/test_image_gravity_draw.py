import unittest
import numpy as np
from .image_gravity_draw import *

class TestImageGravityDraw(unittest.TestCase):
    def test_10000_image_gravity_top_to_bottom(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_draw(input, 0, GravityDrawDirection.TOP_TO_BOTTOM)
        # Assert
        expected = np.array([
            [5, 0, 0, 0, 0],
            [5, 0, 7, 4, 0],
            [5, 0, 7, 4, 0],
            [5, 0, 7, 4, 5],
            [5, 7, 7, 4, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20000_image_gravity_bottom_to_top(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_draw(input, 0, GravityDrawDirection.BOTTOM_TO_TOP)
        # Assert
        expected = np.array([
            [5, 7, 7, 4, 5],
            [0, 7, 7, 4, 5],
            [0, 7, 7, 0, 5],
            [0, 7, 7, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_30000_image_gravity_left_to_right(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_draw(input, 0, GravityDrawDirection.LEFT_TO_RIGHT)
        # Assert
        expected = np.array([
            [5, 5, 5, 5, 5],
            [0, 0, 7, 4, 4],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_40000_image_gravity_right_to_left(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_draw(input, 0, GravityDrawDirection.RIGHT_TO_LEFT)
        # Assert
        expected = np.array([
            [5, 0, 0, 0, 0],
            [7, 7, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [5, 5, 5, 5, 5],
            [7, 7, 7, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
