import unittest
import numpy as np
from .image_erosion import *

class TestImageErode(unittest.TestCase):
    def test_10000_image_erosion_all8(self):
        # Arrange
        image = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]], dtype=np.uint8)
        # Act
        actual = image_erosion(image, ImageErosionId.ALL8)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10001_image_erosion_all8(self):
        # Arrange
        image = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_erosion(image, ImageErosionId.ALL8)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_11000_image_erosion_nearest4(self):
        # Arrange
        image = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
            [1, 0, 1, 1, 1]], dtype=np.uint8)
        # Act
        actual = image_erosion(image, ImageErosionId.NEAREST4)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_11001_image_erosion_nearest4(self):
        # Arrange
        image = np.array([
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1]], dtype=np.uint8)
        # Act
        actual = image_erosion(image, ImageErosionId.NEAREST4)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_12000_image_erosion_corner4(self):
        # Arrange
        image = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]], dtype=np.uint8)
        # Act
        actual = image_erosion(image, ImageErosionId.CORNER4)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

if __name__ == '__main__':
    unittest.main()
