import unittest
import numpy as np
from .image_rect import *
from .rectangle import *

class TestImageRect(unittest.TestCase):
    def test_10000_image_rect_inside_all_of_rect_is_inside(self):
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
        actual = image_rect_inside(image, Rectangle(1, 2, 3, 4), 2)
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

    def test_10001_image_rect_inside_with_big_rect(self):
        # Arrange
        image = np.array([
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9],
            [9, 9, 9, 9, 9]], dtype=np.uint8)
        # Act
        actual = image_rect_inside(image, Rectangle(-10, -10, 30, 30), 2)
        # Assert
        expected = np.array([
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20000_image_rect_outside_center(self):
        # Arrange
        image = np.array([
            [1, 9, 9, 9, 9, 9, 9, 9],
            [9, 1, 9, 9, 9, 9, 9, 9],
            [9, 9, 1, 9, 9, 9, 9, 9],
            [9, 9, 9, 1, 9, 9, 9, 9],
            [9, 9, 9, 9, 1, 9, 9, 5],
            [9, 9, 9, 9, 9, 1, 5, 9],
            [9, 9, 9, 9, 9, 5, 1, 9]], dtype=np.uint8)
        # Act
        actual = image_rect_outside(image, Rectangle(2, 2, 4, 3), 7)
        # Assert
        expected = np.array([
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 1, 9, 9, 9, 7, 7],
            [7, 7, 9, 1, 9, 9, 7, 7],
            [7, 7, 9, 9, 1, 9, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20001_image_rect_outside_left(self):
        # Arrange
        image = np.array([
            [1, 9, 9, 9, 9, 9, 9, 9],
            [9, 1, 9, 9, 9, 9, 9, 9],
            [9, 9, 1, 9, 9, 9, 9, 9],
            [9, 9, 9, 1, 9, 9, 9, 9],
            [9, 9, 9, 9, 1, 9, 9, 5],
            [9, 9, 9, 9, 9, 1, 5, 9],
            [9, 9, 9, 9, 9, 5, 1, 9]], dtype=np.uint8)
        # Act
        actual = image_rect_outside(image, Rectangle(0, 2, 4, 3), 7)
        # Assert
        expected = np.array([
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [9, 9, 1, 9, 7, 7, 7, 7],
            [9, 9, 9, 1, 7, 7, 7, 7],
            [9, 9, 9, 9, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20002_image_rect_outside_bottomleft(self):
        # Arrange
        image = np.array([
            [1, 9, 9, 9, 9, 9, 9, 9],
            [9, 1, 9, 9, 9, 9, 9, 9],
            [9, 9, 1, 9, 9, 9, 9, 9],
            [9, 9, 9, 1, 9, 9, 9, 9],
            [9, 9, 9, 9, 1, 9, 9, 5],
            [9, 9, 9, 9, 9, 1, 5, 9],
            [9, 9, 9, 9, 9, 5, 1, 9]], dtype=np.uint8)
        # Act
        actual = image_rect_outside(image, Rectangle(0, 4, 4, 3), 7)
        # Assert
        expected = np.array([
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [9, 9, 9, 9, 7, 7, 7, 7],
            [9, 9, 9, 9, 7, 7, 7, 7],
            [9, 9, 9, 9, 7, 7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20003_image_rect_outside_bottomleft_clip(self):
        # Arrange
        image = np.array([
            [1, 9, 9, 9, 9, 9, 9, 9],
            [9, 1, 9, 9, 9, 9, 9, 9],
            [9, 9, 1, 9, 9, 9, 9, 9],
            [9, 9, 9, 1, 9, 9, 9, 9],
            [9, 9, 9, 9, 1, 9, 9, 5],
            [9, 9, 9, 9, 9, 1, 5, 9],
            [9, 9, 9, 9, 9, 5, 1, 9]], dtype=np.uint8)
        # Act
        actual = image_rect_outside(image, Rectangle(-1, 4, 5, 4), 7)
        # Assert
        expected = np.array([
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [9, 9, 9, 9, 7, 7, 7, 7],
            [9, 9, 9, 9, 7, 7, 7, 7],
            [9, 9, 9, 9, 7, 7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20004_image_rect_outside_bottomright(self):
        # Arrange
        image = np.array([
            [1, 9, 9, 9, 9, 9, 9, 9],
            [9, 1, 9, 9, 9, 9, 9, 9],
            [9, 9, 1, 9, 9, 9, 9, 9],
            [9, 9, 9, 1, 9, 9, 9, 9],
            [9, 9, 9, 9, 1, 9, 9, 5],
            [9, 9, 9, 9, 9, 1, 5, 9],
            [9, 9, 9, 9, 9, 5, 1, 9]], dtype=np.uint8)
        # Act
        actual = image_rect_outside(image, Rectangle(4, 4, 4, 3), 7)
        # Assert
        expected = np.array([
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 1, 9, 9, 5],
            [7, 7, 7, 7, 9, 1, 5, 9],
            [7, 7, 7, 7, 9, 5, 1, 9]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20005_image_rect_outside_bottomright_clip(self):
        # Arrange
        image = np.array([
            [1, 9, 9, 9, 9, 9, 9, 9],
            [9, 1, 9, 9, 9, 9, 9, 9],
            [9, 9, 1, 9, 9, 9, 9, 9],
            [9, 9, 9, 1, 9, 9, 9, 9],
            [9, 9, 9, 9, 1, 9, 9, 5],
            [9, 9, 9, 9, 9, 1, 5, 9],
            [9, 9, 9, 9, 9, 5, 1, 9]], dtype=np.uint8)
        # Act
        actual = image_rect_outside(image, Rectangle(4, 4, 5, 4), 7)
        # Assert
        expected = np.array([
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 1, 9, 9, 5],
            [7, 7, 7, 7, 9, 1, 5, 9],
            [7, 7, 7, 7, 9, 5, 1, 9]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20006_image_rect_outside_top_clip(self):
        # Arrange
        image = np.array([
            [1, 9, 9, 9, 9, 9, 9, 9],
            [9, 1, 9, 9, 9, 9, 9, 9],
            [9, 9, 1, 9, 9, 9, 9, 9],
            [9, 9, 9, 1, 9, 9, 9, 9],
            [9, 9, 9, 9, 1, 9, 9, 5],
            [9, 9, 9, 9, 9, 1, 5, 9],
            [9, 9, 9, 9, 9, 5, 1, 9]], dtype=np.uint8)
        # Act
        actual = image_rect_outside(image, Rectangle(0, -3, 4, 3), 7)
        # Assert
        expected = np.array([
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7],
            [7, 7, 7, 7, 7, 7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_30000_image_rect_hollow_all_of_rect_is_inside_size1(self):
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

    def test_30001_image_rect_hollow_all_of_rect_is_inside_size2(self):
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
