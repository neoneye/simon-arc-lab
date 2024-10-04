import unittest
import numpy as np
from .image_pixel_similarity import *

class TestImagePixelSimilarity(unittest.TestCase):
    def test_10000_image_pixel_similarity_same_size(self):
        # Arrange
        image0 = np.array([
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2],
            [3, 3, 5, 5, 4, 4, 4],
            [3, 3, 3, 4, 4, 4, 4]], dtype=np.uint8)
        image1 = np.array([
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 5, 5, 2, 2, 2],
            [3, 3, 3, 4, 4, 4, 4],
            [3, 3, 3, 4, 4, 4, 4]], dtype=np.uint8)
        # Act
        actual = image_pixel_similarity(image0, image1)
        # Assert
        expected = {
            1: (8, 9),
            2: (11, 12),
            3: (5, 6),
            4: (7, 8),
            5: (0, 4),
        }
        self.assertEqual(actual, expected)

    def test_10001_image_pixel_similarity_different_size(self):
        # Arrange
        image0 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2],
            [3, 4],
            [5, 6]], dtype=np.uint8)
        # Act
        actual = image_pixel_similarity(image0, image1)
        # Assert
        expected = {}
        self.assertEqual(actual, expected)
