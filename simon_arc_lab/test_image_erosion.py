import unittest
import numpy as np
from .image_erosion import *
from .pixel_connectivity import PixelConnectivity

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
        actual = image_erosion(image, PixelConnectivity.ALL8)
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
        actual = image_erosion(image, PixelConnectivity.ALL8)
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
        actual = image_erosion(image, PixelConnectivity.NEAREST4)
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
        actual = image_erosion(image, PixelConnectivity.NEAREST4)
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
        actual = image_erosion(image, PixelConnectivity.CORNER4)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_13000_image_erosion_lr2(self):
        # Arrange
        image = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]], dtype=np.uint8)
        # Act
        actual = image_erosion(image, PixelConnectivity.LR2)
        # Assert
        expected = np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_14000_image_erosion_tb2(self):
        # Arrange
        image = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]], dtype=np.uint8)
        # Act
        actual = image_erosion(image, PixelConnectivity.TB2)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_15000_image_erosion_tlbr2(self):
        # Arrange
        image = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]], dtype=np.uint8)
        # Act
        actual = image_erosion(image, PixelConnectivity.TLBR2)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_16000_image_erosion_trbl2(self):
        # Arrange
        image = np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0]], dtype=np.uint8)
        # Act
        actual = image_erosion(image, PixelConnectivity.TRBL2)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

if __name__ == '__main__':
    unittest.main()
