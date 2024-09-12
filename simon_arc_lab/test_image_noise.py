import unittest
import numpy as np
from .image_noise import *

class TestImageNoise(unittest.TestCase):
    def test_10000_image_noise_one_pixel_1x1(self):
        # Arrange
        image = np.array([
            [9]], dtype=np.uint8)
        # Act
        actual = image_noise_one_pixel(image, 0)
        # Assert
        expected = np.array([
            [3]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10001_image_noise_one_pixel_3x2(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        actual = image_noise_one_pixel(image, 0)
        # Assert
        expected = np.array([
            [4, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10002_image_noise_one_pixel_3x2(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        actual = image_noise_one_pixel(image, 4)
        # Assert
        expected = np.array([
            [1, 2, 6],
            [4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10003_image_noise_one_pixel_3x2(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        actual = image_noise_one_pixel(image, 9)
        # Assert
        expected = np.array([
            [1, 2, 3],
            [4, 5, 8]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

