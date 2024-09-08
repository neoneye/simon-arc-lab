import unittest
import numpy as np
from .image_paste import *

class TestImagePaste(unittest.TestCase):
    def test_10000_image_paste_at_perfect_fit_no_space_remaining(self):
        # Arrange
        paste_image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        background_image = np.zeros((2, 3), dtype=np.uint8)
        # Act
        actual = image_paste_at(paste_image, background_image, 0, 0)
        # Assert
        expected = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20001_image_paste_at_too_big_cannot_fit_inside(self):
        # Arrange
        paste_image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        background_image = np.zeros((2, 2), dtype=np.uint8)
        # Act
        with self.assertRaises(ValueError):
            image_paste_at(paste_image, background_image, 0, 0)

    def test_20002_image_paste_at_coordinate_outside(self):
        # Arrange
        paste_image = np.array([
            [1, 2],
            [3, 4]], dtype=np.uint8)
        background_image = np.zeros((2, 3), dtype=np.uint8)
        # Act
        with self.assertRaises(ValueError):
            image_paste_at(paste_image, background_image, 2, 0)

    def test_20003_image_paste_at_coordinate_outside(self):
        # Arrange
        paste_image = np.array([
            [1, 2],
            [3, 4]], dtype=np.uint8)
        background_image = np.zeros((3, 2), dtype=np.uint8)
        # Act
        with self.assertRaises(ValueError):
            image_paste_at(paste_image, background_image, 0, 2)

    def test_20000_image_paste_random_perfect_fit_no_space_remaining(self):
        # Arrange
        paste_image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        background_image = np.zeros((2, 3), dtype=np.uint8)
        # Act
        actual = image_paste_random(paste_image, background_image, 0)
        # Assert
        expected = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20001_image_paste_random_too_big_cannot_fit_inside(self):
        # Arrange
        paste_image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        background_image = np.zeros((2, 2), dtype=np.uint8)
        # Act
        with self.assertRaises(ValueError):
            image_paste_random(paste_image, background_image, 0)

    def test_20002_image_paste_random_too_big_cannot_fit_inside(self):
        # Arrange
        paste_image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        background_image = np.zeros((1, 3), dtype=np.uint8)
        # Act
        with self.assertRaises(ValueError):
            image_paste_random(paste_image, background_image, 0)

    def test_21001_image_paste_random_one_x_space(self):
        # Arrange
        paste_image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        background_image = np.zeros((2, 4), dtype=np.uint8)
        # Act
        actual = image_paste_random(paste_image, background_image, 0)
        # Assert
        expected = np.array([
            [1, 2, 3, 0],
            [4, 5, 6, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_21002_image_paste_random_one_x_space(self):
        # Arrange
        paste_image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        background_image = np.zeros((2, 4), dtype=np.uint8)
        # Act
        actual = image_paste_random(paste_image, background_image, 4)
        # Assert
        expected = np.array([
            [0, 1, 2, 3],
            [0, 4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_21003_image_paste_random_two_x_spaces(self):
        # Arrange
        paste_image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        background_image = np.zeros((2, 5), dtype=np.uint8)
        # Act
        actual = image_paste_random(paste_image, background_image, 6)
        # Assert
        expected = np.array([
            [0, 1, 2, 3, 0],
            [0, 4, 5, 6, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_22001_image_paste_random_one_y_space(self):
        # Arrange
        paste_image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        background_image = np.zeros((3, 3), dtype=np.uint8)
        # Act
        actual = image_paste_random(paste_image, background_image, 0)
        # Assert
        expected = np.array([
            [0, 0, 0],
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_22002_image_paste_random_one_y_space(self):
        # Arrange
        paste_image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        background_image = np.zeros((3, 3), dtype=np.uint8)
        # Act
        actual = image_paste_random(paste_image, background_image, 1)
        # Assert
        expected = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_22003_image_paste_random_two_y_spaces(self):
        # Arrange
        paste_image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        background_image = np.zeros((4, 3), dtype=np.uint8)
        # Act
        actual = image_paste_random(paste_image, background_image, 7)
        # Assert
        expected = np.array([
            [0, 0, 0],
            [1, 2, 3],
            [4, 5, 6],
            [0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

