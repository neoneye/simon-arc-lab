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

    def test_50000_image_gravity_topleft_to_bottomright(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_draw(input, 0, GravityDrawDirection.TOPLEFT_TO_BOTTOMRIGHT)
        # Assert
        expected = np.array([
            [5, 0, 0, 0, 0],
            [0, 5, 7, 4, 0],
            [0, 0, 5, 7, 4],
            [0, 0, 0, 5, 5],
            [0, 7, 7, 0, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_60000_image_gravity_topright_to_bottomleft(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_draw(input, 0, GravityDrawDirection.TOPRIGHT_TO_BOTTOMLEFT)
        # Assert
        expected = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 7, 4, 0, 0],
            [7, 4, 0, 0, 5],
            [4, 7, 7, 5, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_70000_image_gravity_bottomleft_to_topright(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_draw(input, 0, GravityDrawDirection.BOTTOMLEFT_TO_TOPRIGHT)
        # Assert
        expected = np.array([
            [5, 0, 0, 7, 4],
            [0, 0, 7, 4, 7],
            [0, 0, 0, 7, 7],
            [0, 0, 7, 7, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_80000_image_gravity_bottomright_to_topleft(self):
        # Arrange
        input = np.array([
            [5, 0, 0, 0, 0],
            [0, 0, 7, 4, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_gravity_draw(input, 0, GravityDrawDirection.BOTTOMRIGHT_TO_TOPLEFT)
        # Assert
        expected = np.array([
            [5, 7, 4, 0, 0],
            [0, 0, 7, 4, 0],
            [7, 0, 0, 5, 0],
            [7, 7, 0, 0, 5],
            [0, 7, 7, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
