import unittest
import numpy as np
from .image_mass_compare import *

class TestImageMassCompare(unittest.TestCase):
    def test_10000_image_mass_compare_adjacent_rows(self):
        # Arrange
        image = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1]], dtype=np.uint8)
        # Act
        actual = image_mass_compare_adjacent_rows(image, 0, 1, 2)
        # Assert
        expected = np.array([
            [1, 1, 1, 1, 1],
            [2, 1, 1, 1, 1],
            [2, 2, 0, 1, 1],
            [2, 2, 2, 2, 1],
            [2, 2, 2, 2, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10001_image_mass_compare_adjacent_rows(self):
        # Arrange
        image = np.array([
            [1, 1, 1, 0, 0],
            [2, 2, 2, 3, 3],
            [4, 5, 4, 5, 4],
            [7, 7, 7, 7, 7]], dtype=np.uint8)
        # Act
        actual = image_mass_compare_adjacent_rows(image, 0, 1, 2)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10002_image_mass_compare_adjacent_rows(self):
        # Arrange
        image = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 3, 3, 0, 0, 0],
            [0, 0, 3, 3, 3, 3, 0, 0],
            [0, 5, 5, 5, 5, 5, 5, 0],
            [0, 0, 3, 3, 3, 3, 0, 0],
            [0, 0, 0, 3, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_mass_compare_adjacent_rows(image, 0, 1, 2)
        # Assert
        expected = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 2, 2, 2, 2, 1, 1],
            [1, 2, 2, 2, 2, 2, 2, 1],
            [2, 1, 1, 1, 1, 1, 1, 2],
            [2, 2, 1, 1, 1, 1, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20000_image_mass_compare_adjacent_columns(self):
        # Arrange
        image = np.array([
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0]], dtype=np.uint8)
        # Act
        actual = image_mass_compare_adjacent_columns(image, 0, 1, 2)
        # Assert
        expected = np.array([
            [1, 2, 2, 2, 2],
            [1, 1, 2, 2, 2],
            [1, 1, 0, 2, 2],
            [1, 1, 1, 2, 2],
            [1, 1, 1, 1, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20001_image_mass_compare_adjacent_columns(self):
        # Arrange
        image = np.array([
            [1, 2, 4, 7],
            [1, 2, 5, 7],
            [1, 2, 4, 7],
            [0, 3, 5, 7],
            [0, 3, 4, 7]], dtype=np.uint8)
        # Act
        actual = image_mass_compare_adjacent_columns(image, 0, 1, 2)
        # Assert
        expected = np.array([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

if __name__ == '__main__':
    unittest.main()
