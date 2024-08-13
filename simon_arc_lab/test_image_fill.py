import unittest
import numpy as np
from .image_fill import *

class TestImageFill(unittest.TestCase):
    def test_10000_flood_fill_nearest4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 0, 0, 5, 3, PixelConnectivity.NEAREST4)
        # Assert
        expected = np.array([
            [3, 3, 3, 3, 3],
            [3, 8, 8, 3, 8],
            [3, 8, 3, 3, 8],
            [3, 3, 3, 3, 8]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_10001_flood_fill_nearest4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 1, 1, 8, 1, PixelConnectivity.NEAREST4)
        # Assert
        expected = np.array([
            [5, 5, 5, 5, 5],
            [5, 1, 1, 5, 8],
            [5, 1, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_10002_flood_fill_nearest4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 4, 1, 8, 1, PixelConnectivity.NEAREST4)
        # Assert
        expected = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 1],
            [5, 8, 5, 5, 1],
            [5, 5, 5, 5, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_10003_flood_fill_nearest4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 2, 1, 0, 0, PixelConnectivity.NEAREST4)
        # Assert
        expected = np.array([
            [5, 5, 5, 5, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_11000_flood_fill_all8(self):
        # Arrange
        image = np.array([
            [5, 3, 3, 3, 3, 5],
            [3, 5, 3, 5, 3, 3],
            [3, 3, 5, 3, 5, 3],
            [5, 3, 3, 3, 3, 5]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 3, 1, 5, 0, PixelConnectivity.ALL8)
        # Assert
        expected = np.array([
            [0, 3, 3, 3, 3, 5],
            [3, 0, 3, 0, 3, 3],
            [3, 3, 0, 3, 0, 3],
            [5, 3, 3, 3, 3, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_11001_flood_fill_all8(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 2, 1, 0, 0, PixelConnectivity.ALL8)
        # Assert
        expected = np.array([
            [5, 5, 5, 5, 5],
            [5, 0, 0, 0, 5],
            [5, 0, 0, 0, 5],
            [5, 5, 5, 5, 5]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_12000_flood_fill_corner4(self):
        # Arrange
        image = np.array([
            [3, 7, 3, 3, 3, 3],
            [7, 3, 7, 3, 3, 3],
            [3, 3, 3, 7, 3, 7],
            [3, 3, 3, 3, 7, 3]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 2, 1, 7, 9, PixelConnectivity.CORNER4)
        # Assert
        expected = np.array([
            [3, 9, 3, 3, 3, 3],
            [9, 3, 9, 3, 3, 3],
            [3, 3, 3, 9, 3, 9],
            [3, 3, 3, 3, 9, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_12001_flood_fill_corner4(self):
        # Arrange
        image = np.array([
            [7, 7, 7, 3, 3, 3],
            [3, 3, 7, 3, 3, 3],
            [3, 3, 7, 3, 3, 7],
            [3, 3, 7, 7, 7, 7]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 2, 1, 7, 9, PixelConnectivity.CORNER4)
        # Assert
        expected = np.array([
            [7, 9, 7, 3, 3, 3],
            [3, 3, 9, 3, 3, 3],
            [3, 3, 7, 3, 3, 7],
            [3, 3, 7, 7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_13000_flood_fill_lr2(self):
        # Arrange
        image = np.array([
            [7, 7, 7, 3, 3, 3],
            [3, 3, 7, 3, 3, 3],
            [3, 3, 7, 3, 3, 7],
            [3, 3, 7, 7, 7, 7]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 4, 1, 3, 9, PixelConnectivity.LR2)
        # Assert
        expected = np.array([
            [7, 7, 7, 3, 3, 3],
            [3, 3, 7, 9, 9, 9],
            [3, 3, 7, 3, 3, 7],
            [3, 3, 7, 7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_14000_flood_fill_tb2(self):
        # Arrange
        image = np.array([
            [7, 7, 7, 3, 3, 3],
            [3, 3, 7, 3, 3, 3],
            [3, 3, 7, 3, 3, 7],
            [3, 3, 7, 7, 7, 7]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 4, 1, 3, 9, PixelConnectivity.TB2)
        # Assert
        expected = np.array([
            [7, 7, 7, 3, 9, 3],
            [3, 3, 7, 3, 9, 3],
            [3, 3, 7, 3, 9, 7],
            [3, 3, 7, 7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_15000_flood_fill_tlbr2(self):
        # Arrange
        image = np.array([
            [7, 7, 7, 3, 3, 3],
            [3, 3, 7, 3, 3, 3],
            [3, 3, 7, 3, 3, 3],
            [3, 3, 7, 7, 7, 7]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 4, 1, 3, 9, PixelConnectivity.TLBR2)
        # Assert
        expected = np.array([
            [7, 7, 7, 9, 3, 3],
            [3, 3, 7, 3, 9, 3],
            [3, 3, 7, 3, 3, 9],
            [3, 3, 7, 7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_16000_flood_fill_trbl2(self):
        # Arrange
        image = np.array([
            [7, 7, 7, 3, 3, 3],
            [3, 3, 7, 3, 3, 3],
            [3, 3, 7, 3, 3, 3],
            [3, 3, 7, 7, 7, 7]], dtype=np.uint8)
        # Act
        image_flood_fill(image, 4, 1, 3, 9, PixelConnectivity.TRBL2)
        # Assert
        expected = np.array([
            [7, 7, 7, 3, 3, 9],
            [3, 3, 7, 3, 9, 3],
            [3, 3, 7, 9, 3, 3],
            [3, 3, 7, 7, 7, 7]], dtype=np.uint8)
        np.testing.assert_array_equal(image, expected)

    def test_30000_mask_flood_fill_nearest4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        output = np.zeros((4, 5), dtype=np.uint8)
        color = image[0, 0]

        # Act
        image_mask_flood_fill(output, image, 0, 0, color, PixelConnectivity.NEAREST4)

        # Assert
        expected = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 1, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

    def test_30001_mask_flood_fill_nearest4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        output = np.zeros((4, 5), dtype=np.uint8)
        color = image[1, 1]

        # Act
        image_mask_flood_fill(output, image, 1, 1, color, PixelConnectivity.NEAREST4)

        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

    def test_30002_mask_flood_fill_nearest4(self):
        # Arrange
        image = np.array([
            [5, 5, 5, 5, 5],
            [5, 8, 8, 5, 8],
            [5, 8, 5, 5, 8],
            [5, 5, 5, 5, 8]], dtype=np.uint8)
        output = np.zeros((4, 5), dtype=np.uint8)
        color = image[1, 4]

        # Act
        image_mask_flood_fill(output, image, 4, 1, color, PixelConnectivity.NEAREST4)

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
        image_mask_flood_fill(output, image, 2, 0, color, PixelConnectivity.NEAREST4)

        # Assert
        expected = np.array([
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

    def test_31000_mask_flood_fill_all8(self):
        # Arrange
        image = np.array([
            [9, 5, 5],
            [5, 9, 5],
            [5, 5, 9]], dtype=np.uint8)
        output = np.zeros((3, 3), dtype=np.uint8)
        color = image[0, 2]

        # Act
        image_mask_flood_fill(output, image, 2, 0, color, PixelConnectivity.ALL8)

        # Assert
        expected = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

    def test_32000_mask_flood_fill_corner4(self):
        # Arrange
        image = np.array([
            [5, 8, 8, 5, 5],
            [8, 5, 8, 8, 5],
            [8, 8, 5, 8, 8],
            [8, 8, 8, 5, 8]], dtype=np.uint8)
        output = np.zeros((4, 5), dtype=np.uint8)
        color = image[0, 0]

        # Act
        image_mask_flood_fill(output, image, 0, 0, color, PixelConnectivity.CORNER4)

        # Assert
        expected = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

    def test_32001_mask_flood_fill_corner4(self):
        # Arrange
        image = np.array([
            [5, 8, 8, 8, 5],
            [8, 5, 5, 5, 8],
            [8, 5, 8, 5, 8],
            [8, 5, 5, 5, 8]], dtype=np.uint8)
        output = np.zeros((4, 5), dtype=np.uint8)
        color = image[0, 4]

        # Act
        image_mask_flood_fill(output, image, 4, 0, color, PixelConnectivity.CORNER4)

        # Assert
        expected = np.array([
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

    def test_33000_mask_flood_fill_lr2(self):
        # Arrange
        image = np.array([
            [5, 8, 8, 8, 5],
            [8, 5, 5, 5, 8],
            [8, 5, 8, 5, 8],
            [8, 5, 5, 5, 8]], dtype=np.uint8)
        output = np.zeros((4, 5), dtype=np.uint8)
        color = 5

        # Act
        image_mask_flood_fill(output, image, 2, 1, color, PixelConnectivity.LR2)

        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

    def test_34000_mask_flood_fill_tb2(self):
        # Arrange
        image = np.array([
            [5, 8, 8, 8, 5],
            [8, 5, 5, 5, 8],
            [8, 5, 8, 5, 8],
            [8, 5, 5, 5, 8]], dtype=np.uint8)
        output = np.zeros((4, 5), dtype=np.uint8)
        color = 5

        # Act
        image_mask_flood_fill(output, image, 1, 2, color, PixelConnectivity.TB2)

        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

    def test_35000_mask_flood_fill_tlbr2(self):
        # Arrange
        image = np.array([
            [5, 8, 8, 8, 5],
            [8, 5, 5, 5, 8],
            [8, 5, 5, 5, 8],
            [8, 5, 5, 5, 8]], dtype=np.uint8)
        output = np.zeros((4, 5), dtype=np.uint8)
        color = 5

        # Act
        image_mask_flood_fill(output, image, 2, 2, color, PixelConnectivity.TLBR2)

        # Assert
        expected = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

    def test_36000_mask_flood_fill_trbl2(self):
        # Arrange
        image = np.array([
            [5, 8, 8, 8, 5],
            [8, 5, 5, 5, 8],
            [8, 5, 5, 5, 8],
            [8, 5, 5, 5, 8]], dtype=np.uint8)
        output = np.zeros((4, 5), dtype=np.uint8)
        color = 5

        # Act
        image_mask_flood_fill(output, image, 2, 2, color, PixelConnectivity.TRBL2)

        # Assert
        expected = np.array([
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)

if __name__ == '__main__':
    unittest.main()
