import unittest
import numpy as np
from .image_fill import *

class TestImageFill(unittest.TestCase):
    def test_10000_flood_fill4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        # Act
        image_fill_inplace(image, 0, 0, 5, 3, PixelConnectivity.CONNECTIVITY4)
        # Assert
        expected = np.array([
            [3, 3, 3, 3, 3],
            [3, 8, 8, 3, 8],
            [3, 8, 3, 3, 8],
            [3, 3, 3, 3, 8]], dtype=np.uint8)
        self.assertTrue(np.array_equal(image, expected))

    def test_10001_flood_fill4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        # Act
        image_fill_inplace(image, 1, 1, 8, 1, PixelConnectivity.CONNECTIVITY4)
        # Assert
        expected = np.array([
            [5, 5, 5, 5, 5],
            [5, 1, 1, 5, 8],
            [5, 1, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        self.assertTrue(np.array_equal(image, expected))

    def test_10002_flood_fill4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        # Act
        image_fill_inplace(image, 4, 1, 8, 1, PixelConnectivity.CONNECTIVITY4)
        # Assert
        expected = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 1],
            [5, 8, 5, 5, 1],
            [5, 5, 5, 5, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(image, expected))

    def test_10003_flood_fill4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        # Act
        image_fill_inplace(image, 2, 1, 0, 0, PixelConnectivity.CONNECTIVITY4)
        # Assert
        expected = np.array([
            [5, 5, 5, 5, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        self.assertTrue(np.array_equal(image, expected))

    def test_20000_flood_fill8(self):
        # Arrange
        image = np.array([
            [5, 3, 3, 3, 3, 5],
            [3, 5, 3, 5, 3, 3],
            [3, 3, 5, 3, 5, 3],
            [5, 3, 3, 3, 3, 5]], dtype=np.uint8)
        # Act
        image_fill_inplace(image, 3, 1, 5, 0, PixelConnectivity.CONNECTIVITY8)
        # Assert
        expected = np.array([
            [0, 3, 3, 3, 3, 5],
            [3, 0, 3, 0, 3, 3],
            [3, 3, 0, 3, 0, 3],
            [5, 3, 3, 3, 3, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(image, expected))

    def test_20001_flood_fill8(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        # Act
        image_fill_inplace(image, 2, 1, 0, 0, PixelConnectivity.CONNECTIVITY8)
        # Assert
        expected = np.array([
            [5, 5, 5, 5, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        self.assertTrue(np.array_equal(image, expected))

if __name__ == '__main__':
    unittest.main()
