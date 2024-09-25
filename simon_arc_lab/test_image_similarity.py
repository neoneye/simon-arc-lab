import unittest
import numpy as np
from .image_similarity import *

class TestImageSimilarity(unittest.TestCase):
    def test_10000_compute_jaccard_index(self):
        self.assertEqual(ImageSimilarity.compute_jaccard_index([False, False]), 0)
        self.assertEqual(ImageSimilarity.compute_jaccard_index([False, True]), 50)
        self.assertEqual(ImageSimilarity.compute_jaccard_index([True, False]), 50)
        self.assertEqual(ImageSimilarity.compute_jaccard_index([True, True]), 100)

    def test_20000_same_histogram_true(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [3, 6],
            [2, 5],
            [1, 4]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_histogram()
        # Assert
        self.assertEqual(actual, True)

    def test_20001_same_histogram_false(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [3, 3],
            [3, 3],
            [3, 3]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_histogram()
        # Assert
        self.assertEqual(actual, False)

    def test_21000_same_unique_colors_true(self):
        # Arrange
        image0 = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_unique_colors()
        # Assert
        self.assertEqual(actual, True)

    def test_21001_same_histogram_false(self):
        # Arrange
        image0 = np.array([
            [3, 4, 5],
            [6, 7, 8]], dtype=np.uint8)
        image1 = np.array([
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6]], dtype=np.uint8)
        i = ImageSimilarity(image0, image1)
        # Act
        actual = i.same_unique_colors()
        # Assert
        self.assertEqual(actual, False)
