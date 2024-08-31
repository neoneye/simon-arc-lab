import unittest
import numpy as np
from .image_gravity_move import *

class TestImageGravityMove(unittest.TestCase):
    def test_10000_image_gravity_left(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_move(input, 0, GravityMoveDirection.LEFT)
        # Assert
        expected = np.array([
            [5, 0, 0, 0, 0],
            [7, 4, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 0],
            [7, 7, 0, 0, 0]], dtype=np.uint8)
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
        actual = image_gravity_move(input, 0, GravityMoveDirection.UP)
        # Assert
        expected = np.array([
            [5, 7, 7, 4, 5],
            [0, 0, 7, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
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
        actual = image_gravity_move(input, 0, GravityMoveDirection.DOWN)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 7, 0, 0],
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
        actual = image_gravity_move(input, 0, GravityMoveDirection.RIGHT)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 5],
            [0, 0, 0, 7, 4],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 0, 0, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
