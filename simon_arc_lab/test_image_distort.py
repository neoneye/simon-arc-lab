import unittest
import numpy as np
from .image_distort import *

class TestImageDistort(unittest.TestCase):
    def test_10000_image_distort_iteration1_impact10percent(self):
        # Arrange
        image = np.array([
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [4, 4, 4, 5, 5, 5, 6, 6, 6],
            [4, 4, 4, 5, 5, 5, 6, 6, 6],
            [4, 4, 4, 5, 5, 5, 6, 6, 6]], dtype=np.uint8)
        # Act
        actual = image_distort(image, 1, 10, 0)
        # Assert
        expected = np.array([
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 1, 1, 2, 5, 2, 3, 3, 3],
            [4, 4, 4, 5, 2, 5, 6, 6, 6],
            [4, 4, 4, 5, 5, 5, 6, 6, 6],
            [4, 4, 4, 5, 5, 6, 5, 6, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10001_image_distort_iteration1_impact30percent(self):
        # Arrange
        image = np.array([
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [4, 4, 4, 5, 5, 5, 6, 6, 6],
            [4, 4, 4, 5, 5, 5, 6, 6, 6],
            [4, 4, 4, 5, 5, 5, 6, 6, 6]], dtype=np.uint8)
        # Act
        actual = image_distort(image, 1, 30, 0)
        # Assert
        expected = np.array([
            [1, 1, 2, 2, 5, 3, 2, 3, 3],
            [1, 1, 1, 1, 2, 2, 3, 3, 3],
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [4, 4, 5, 4, 2, 5, 6, 6, 6],
            [4, 4, 4, 4, 5, 5, 6, 5, 6],
            [4, 5, 4, 5, 5, 6, 6, 6, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10002_image_distort_iteration1_impact100percent(self):
        # Arrange
        image = np.array([
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [4, 4, 4, 5, 5, 5, 6, 6, 6],
            [4, 4, 4, 5, 5, 5, 6, 6, 6],
            [4, 4, 4, 5, 5, 5, 6, 6, 6]], dtype=np.uint8)
        # Act
        actual = image_distort(image, 1, 100, 0)
        # Assert
        expected = np.array([
            [1, 1, 4, 1, 6, 2, 3, 3, 5],
            [1, 2, 1, 4, 3, 1, 3, 3, 2],
            [5, 4, 2, 2, 2, 5, 5, 6, 3],
            [4, 4, 5, 1, 5, 3, 2, 6, 5],
            [1, 1, 2, 4, 3, 6, 6, 3, 2],
            [5, 4, 4, 5, 6, 4, 6, 6, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10003_image_distort_iteration5_impact10percent(self):
        # Arrange
        image = np.array([
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [4, 4, 4, 5, 5, 5, 6, 6, 6],
            [4, 4, 4, 5, 5, 5, 6, 6, 6],
            [4, 4, 4, 5, 5, 5, 6, 6, 6]], dtype=np.uint8)
        # Act
        actual = image_distort(image, 5, 10, 0)
        # Assert
        expected = np.array([
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 1, 1, 2, 5, 2, 3, 3, 3],
            [1, 1, 4, 2, 2, 2, 3, 6, 3],
            [4, 4, 1, 5, 5, 2, 6, 3, 6],
            [4, 4, 4, 5, 5, 5, 6, 6, 6],
            [4, 4, 4, 5, 5, 6, 5, 6, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
