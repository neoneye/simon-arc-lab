import unittest
import numpy as np
from .image_gravity_move import *

class TestImageGravityMove(unittest.TestCase):
    def test_10000_image_gravity_move_left(self):
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

    def test_10001_image_gravity_move_left(self):
        # Arrange
        input = np.array([
            [1, 1, 2, 2, 3],
            [0, 0, 4, 4, 4],
            [5, 0, 5, 0, 5]], dtype=np.uint8)
        # Act
        actual = image_gravity_move(input, 0, GravityMoveDirection.LEFT)
        # Assert
        expected = np.array([
            [1, 1, 2, 2, 3],
            [4, 4, 4, 0, 0],
            [5, 5, 5, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20000_image_gravity_move_up(self):
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

    def test_30000_image_gravity_move_down(self):
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

    def test_40000_image_gravity_move_right(self):
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

    def test_50000_image_gravity_move_topleft_to_bottomright(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 9],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_move(input, 0, GravityMoveDirection.TOPLEFT_TO_BOTTOMRIGHT)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 9],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 7, 4],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_50001_image_gravity_move_topleft_to_bottomright(self):
        # Arrange
        input = np.array([
            [4, 0, 1, 0, 6],
            [0, 0, 5, 2, 0],
            [0, 0, 0, 0, 3]], dtype=np.uint8)
        # Act
        actual = image_gravity_move(input, 0, GravityMoveDirection.TOPLEFT_TO_BOTTOMRIGHT)
        # Assert
        expected = np.array([
            [0, 0, 1, 0, 6],
            [0, 0, 0, 2, 0],
            [0, 0, 4, 5, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_60000_image_gravity_move_bottomright_to_topleft(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 9],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_move(input, 0, GravityMoveDirection.BOTTOMRIGHT_TO_TOPLEFT)
        # Assert
        expected = np.array([
            [5, 7, 4, 0, 9],
            [0, 0, 5, 0, 0],
            [7, 0, 0, 0, 0],
            [7, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_70000_image_gravity_move_topright_to_bottomleft(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 9],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_move(input, 0, GravityMoveDirection.TOPRIGHT_TO_BOTTOMLEFT)
        # Assert
        expected = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [7, 9, 0, 0, 0],
            [4, 7, 7, 5, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_80000_image_gravity_move_bottomleft_to_topright(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 9],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_move(input, 0, GravityMoveDirection.BOTTOMLEFT_TO_TOPRIGHT)
        # Assert
        expected = np.array([
            [5, 0, 0, 7, 9],
            [0, 0, 0, 4, 7],
            [0, 0, 0, 0, 7],
            [0, 0, 0, 0, 5],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
