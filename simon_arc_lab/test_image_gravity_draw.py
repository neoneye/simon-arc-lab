import unittest
import numpy as np
from .image_gravity_draw import *

class TestImageGravityDraw(unittest.TestCase):
    def test_10000_image_gravity_left(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_draw(input, 0, GravityDrawDirection.LEFT)
        # Assert
        expected = np.array([
            [5, 0, 0, 0, 0],
            [7, 7, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [5, 5, 5, 5, 5],
            [7, 7, 7, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10001_image_gravity_up(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_draw(input, 0, GravityDrawDirection.UP)
        # Assert
        expected = np.array([
            [5, 7, 7, 4, 5],
            [0, 7, 7, 4, 5],
            [0, 7, 7, 0, 5],
            [0, 7, 7, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10002_image_gravity_down(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_draw(input, 0, GravityDrawDirection.DOWN)
        # Assert
        expected = np.array([
            [5, 0, 0, 0, 0],
            [5, 0, 7, 4, 0],
            [5, 0, 7, 4, 0],
            [5, 0, 7, 4, 5],
            [5, 7, 7, 4, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10003_image_gravity_right(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_draw(input, 0, GravityDrawDirection.RIGHT)
        # Assert
        expected = np.array([
            [5, 5, 5, 5, 5],
            [0, 0, 7, 4, 4],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
