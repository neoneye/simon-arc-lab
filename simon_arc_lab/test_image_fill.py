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
        image_flood_fill(image, 0, 0, 5, 3, PixelConnectivity.CONNECTIVITY4)
        # Assert
        expected = np.array([
            [3, 3, 3, 3, 3],
            [3, 8, 8, 3, 8],
            [3, 8, 3, 3, 8],
            [3, 3, 3, 3, 8]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_10001_flood_fill4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 1, 1, 8, 1, PixelConnectivity.CONNECTIVITY4)
        # Assert
        expected = np.array([
            [5, 5, 5, 5, 5],
            [5, 1, 1, 5, 8],
            [5, 1, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_10002_flood_fill4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 4, 1, 8, 1, PixelConnectivity.CONNECTIVITY4)
        # Assert
        expected = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 1],
            [5, 8, 5, 5, 1],
            [5, 5, 5, 5, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_10003_flood_fill4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 2, 1, 0, 0, PixelConnectivity.CONNECTIVITY4)
        # Assert
        expected = np.array([
            [5, 5, 5, 5, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_11000_flood_fill8(self):
        # Arrange
        image = np.array([
            [5, 3, 3, 3, 3, 5],
            [3, 5, 3, 5, 3, 3],
            [3, 3, 5, 3, 5, 3],
            [5, 3, 3, 3, 3, 5]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 3, 1, 5, 0, PixelConnectivity.CONNECTIVITY8)
        # Assert
        expected = np.array([
            [0, 3, 3, 3, 3, 5],
            [3, 0, 3, 0, 3, 3],
            [3, 3, 0, 3, 0, 3],
            [5, 3, 3, 3, 3, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_11001_flood_fill8(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 2, 1, 0, 0, PixelConnectivity.CONNECTIVITY8)
        # Assert
        expected = np.array([
            [5, 5, 5, 5, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_12000_flood_fill4diagonal(self):
        # Arrange
        image = np.array([
            [3, 7, 3, 3, 3, 3],
            [7, 3, 7, 3, 3, 3],
            [3, 3, 3, 7, 3, 7],
            [3, 3, 3, 3, 7, 3]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 2, 1, 7, 9, PixelConnectivity.CONNECTIVITY4DIAGONAL)
        # Assert
        expected = np.array([
            [3, 9, 3, 3, 3, 3],
            [9, 3, 9, 3, 3, 3],
            [3, 3, 3, 9, 3, 9],
            [3, 3, 3, 3, 9, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_12001_flood_fill4diagonal(self):
        # Arrange
        image = np.array([
            [7, 7, 7, 3, 3, 3],
            [3, 3, 7, 3, 3, 3],
            [3, 3, 7, 3, 3, 7],
            [3, 3, 7, 7, 7, 7]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 2, 1, 7, 9, PixelConnectivity.CONNECTIVITY4DIAGONAL)
        # Assert
        expected = np.array([
            [7, 9, 7, 3, 3, 3],
            [3, 3, 9, 3, 3, 3],
            [3, 3, 7, 3, 3, 7],
            [3, 3, 7, 7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_30000_mask_flood_fill4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        output = np.zeros((4, 5), dtype=np.uint8)
        color = image[0, 0]

        # Act
        image_mask_flood_fill(output, image, 0, 0, color, PixelConnectivity.CONNECTIVITY4)

        # Assert
        expected = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 1, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

    def test_30001_mask_flood_fill4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        output = np.zeros((4, 5), dtype=np.uint8)
        color = image[1, 1]

        # Act
        image_mask_flood_fill(output, image, 1, 1, color, PixelConnectivity.CONNECTIVITY4)

        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

    def test_30002_mask_flood_fill4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        output = np.zeros((4, 5), dtype=np.uint8)
        color = image[1, 4]

        # Act
        image_mask_flood_fill(output, image, 4, 1, color, PixelConnectivity.CONNECTIVITY4)

        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

    def test_30003_mask_flood_fill4(self):
        # Arrange
        image = np.array([
            [9, 5, 5],
            [5, 9, 5],
            [5, 5, 9]], dtype=np.uint8)
        output = np.zeros((3, 3), dtype=np.uint8)
        color = image[0, 2]

        # Act
        image_mask_flood_fill(output, image, 2, 0, color, PixelConnectivity.CONNECTIVITY4)

        # Assert
        expected = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

    def test_40000_mask_flood_fill8(self):
        # Arrange
        image = np.array([
            [9, 5, 5],
            [5, 9, 5],
            [5, 5, 9]], dtype=np.uint8)
        output = np.zeros((3, 3), dtype=np.uint8)
        color = image[0, 2]

        # Act
        image_mask_flood_fill(output, image, 2, 0, color, PixelConnectivity.CONNECTIVITY8)

        # Assert
        expected = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

if __name__ == '__main__':
    unittest.main()
