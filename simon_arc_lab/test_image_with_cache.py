import unittest
import numpy as np
from .image_with_cache import ImageWithCache

class TestImageWithCache(unittest.TestCase):
    def test_10000_histogram(self):
        # Arrange
        input = np.array([
            [1, 1],
            [2, 3],
            [3, 3]], dtype=np.uint8)
        image = ImageWithCache(input)
        # Act
        actual = image.histogram().pretty()
        # Assert
        expected = '3:3,1:2,2:1'
        self.assertEqual(actual, expected)

    def test_20000_bigrams_direction_all(self):
        # Arrange
        input = np.array([
            [1, 1],
            [2, 3],
            [3, 3]], dtype=np.uint8)
        image = ImageWithCache(input)
        # Act
        actual = image.bigrams_direction_all()
        # Assert
        expected = [(1, 2), (1, 3), (1, 255), (2, 3), (2, 255), (3, 255)]
        self.assertEqual(actual, expected)

    def test_30000_shape2x2_id_list(self):
        # Arrange
        input = np.array([
            [1, 1],
            [2, 3],
            [3, 3]], dtype=np.uint8)
        image = ImageWithCache(input)
        # Act
        actual = image.shape2x2_id_list()
        # Assert
        expected = [1, 3, 4, 8, 12, 22, 25, 37, 42]
        self.assertEqual(actual, expected)
