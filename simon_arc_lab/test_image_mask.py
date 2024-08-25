import unittest
import numpy as np
from .image_mask import *

class TestImageMask(unittest.TestCase):
    def test_10000_image_mask_and(self):
        # Arrange
        a = np.array([
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0]], dtype=np.uint8)
        b = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_mask_and(a, b)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_11000_image_mask_or(self):
        # Arrange
        a = np.array([
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0]], dtype=np.uint8)
        b = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_mask_or(a, b)
        # Assert
        expected = np.array([
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_12000_image_mask_xor(self):
        # Arrange
        a = np.array([
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0]], dtype=np.uint8)
        b = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        # Act
        actual = image_mask_xor(a, b)
        # Assert
        expected = np.array([
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 1],
            [0, 1, 1, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)
