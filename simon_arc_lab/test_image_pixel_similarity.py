import unittest
import numpy as np
from .image_pixel_similarity import *

class TestImagePixelSimilarity(unittest.TestCase):
    def test_10000_image_pixel_similarity_dict_same_size(self):
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
        actual = image_pixel_similarity_dict(image0, image1)
        # Assert
        expected = {
            1: (8, 9),
            2: (11, 12),
            3: (5, 6),
            4: (7, 8),
            5: (0, 4),
        }
        self.assertEqual(actual, expected)

    def test_10001_image_pixel_similarity_dict_different_size(self):
        # Arrange
        image0 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2],
            [3, 4],
            [5, 6]], dtype=np.uint8)
        # Act
        actual = image_pixel_similarity_dict(image0, image1)
        # Assert
        expected = {}
        self.assertEqual(actual, expected)

    def test_20000_jaccard_index_from_image_pixel_similarity_dict_empty(self):
        # Arrange
        dict = {}
        # Act
        actual = jaccard_index_from_image_pixel_similarity_dict(dict)
        # Assert
        expected = 100
        self.assertEqual(actual, expected)

    def test_20001_jaccard_index_from_image_pixel_similarity_dict_2items_identical(self):
        # Arrange
        dict = {
            1: (100, 100),
            2: (200, 200),
        }
        # Act
        actual = jaccard_index_from_image_pixel_similarity_dict(dict)
        # Assert
        expected = 100
        self.assertEqual(actual, expected)

    def test_20002_jaccard_index_from_image_pixel_similarity_dict_2items(self):
        # Arrange
        dict = {
            1: (50, 100),
            2: (50, 100),
        }
        # Act
        actual = jaccard_index_from_image_pixel_similarity_dict(dict)
        # Assert
        expected = 50
        self.assertEqual(actual, expected)

    def test_20003_jaccard_index_from_image_pixel_similarity_dict_3items(self):
        # Arrange
        dict = {
            1: (50, 100),
            2: (25, 100),
            3: (75, 100),
        }
        # Act
        actual = jaccard_index_from_image_pixel_similarity_dict(dict)
        # Assert
        expected = 50
        self.assertEqual(actual, expected)

    def test_30000_image_pixel_similarity_jaccard_index_identical(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        actual = image_pixel_similarity_jaccard_index(image, image)
        # Assert
        self.assertEqual(actual, 100)

    def test_30001_image_pixel_similarity_jaccard_index_nothing_in_common(self):
        # Arrange
        image0 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        image1 = np.array([
            [7, 7, 7],
            [8, 8, 8]], dtype=np.uint8)
        # Act
        actual = image_pixel_similarity_jaccard_index(image0, image1)
        # Assert
        self.assertEqual(actual, 0)

    def test_30002_image_pixel_similarity_jaccard_index_somewhat_similar(self):
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
        actual = image_pixel_similarity_jaccard_index(image0, image1)
        # Assert
        expected = 100 * (8 + 11 + 5 + 7) // (9 + 12 + 6 + 8 + 4)
        self.assertEqual(expected, 79)
        self.assertEqual(actual, expected)

    def test_30003_image_pixel_similarity_jaccard_index_different_sizes(self):
        # Arrange
        image0 = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2],
            [3, 4],
            [5, 6]], dtype=np.uint8)
        # Act
        actual = image_pixel_similarity_jaccard_index(image0, image1)
        # Assert
        self.assertEqual(actual, 0)

