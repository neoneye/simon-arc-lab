import unittest
import numpy as np
from .image_gravity_move import *

class TestImageGravityMove(unittest.TestCase):
    def test_10000_image_gravity_move_top_to_bottom(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_move(input, 0, GravityMoveDirection.TOP_TO_BOTTOM)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 7, 0, 0],
            [5, 7, 7, 4, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20000_image_gravity_move_bottom_to_top(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_move(input, 0, GravityMoveDirection.BOTTOM_TO_TOP)
        # Assert
        expected = np.array([
            [5, 7, 7, 4, 5],
            [0, 0, 7, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_30000_image_gravity_move_left_to_right(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_move(input, 0, GravityMoveDirection.LEFT_TO_RIGHT)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 5],
            [0, 0, 0, 7, 4],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 0, 0, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_40000_image_gravity_move_right_to_left(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_move(input, 0, GravityMoveDirection.RIGHT_TO_LEFT)
        # Assert
        expected = np.array([
            [5, 0, 0, 0, 0],
            [7, 4, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [5, 0, 0, 0, 0],
            [7, 7, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_40001_image_gravity_move_right_to_left(self):
        # Arrange
        input = np.array([
            [1, 1, 2, 2, 3],
            [0, 0, 4, 4, 4],
            [5, 0, 5, 0, 5]], dtype=np.uint8)
        # Act
        actual = image_gravity_move(input, 0, GravityMoveDirection.RIGHT_TO_LEFT)
        # Assert
        expected = np.array([
            [1, 1, 2, 2, 3],
            [4, 4, 4, 0, 0],
            [5, 5, 5, 0, 0]], dtype=np.uint8)
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

    def test_90000_image_collapse_color_horizontal(self):
        # Arrange
        input = np.array([[5, 0, 6, 0, 7]], dtype=np.uint8)
        # Act
        actual = image_collapse_color(input, 0)
        # Assert
        expected = np.array([[5, 6, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_90001_image_collapse_color_horizontal(self):
        # Arrange
        input = np.array([
            [5, 0, 6, 0, 7],
            [3, 0, 2, 0, 1]], dtype=np.uint8)
        # Act
        actual = image_collapse_color(input, 0)
        # Assert
        expected = np.array([[5, 6, 7], [3, 2, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_90002_image_collapse_color_vertical(self):
        # Arrange
        input = np.array([
            [5], 
            [0], 
            [6], 
            [0], 
            [7]], dtype=np.uint8)
        # Act
        actual = image_collapse_color(input, 0)
        # Assert
        expected = np.array([
            [5], 
            [6], 
            [7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_90003_image_collapse_color_vertical(self):
        # Arrange
        input = np.array([
            [5, 3], 
            [0, 0], 
            [6, 2], 
            [0, 0], 
            [7, 1]], dtype=np.uint8)
        # Act
        actual = image_collapse_color(input, 0)
        # Assert
        expected = np.array([
            [5, 3], 
            [6, 2], 
            [7, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_90004_image_collapse_color_both_horizontal_and_vertical(self):
        # Arrange
        input = np.array([
            [5, 0, 3, 0, 8], 
            [0, 0, 0, 0, 0], 
            [6, 0, 2, 0, 8], 
            [0, 0, 0, 0, 0], 
            [7, 0, 1, 0, 8]], dtype=np.uint8)
        # Act
        actual = image_collapse_color(input, 0)
        # Assert
        expected = np.array([
            [5, 3, 8], 
            [6, 2, 8], 
            [7, 1, 8]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
