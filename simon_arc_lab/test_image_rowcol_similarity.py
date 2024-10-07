import unittest
import numpy as np
from .image_rowcol_similarity import *

class TestImageRowColSimilarity(unittest.TestCase):
    def test_10000_image_transition_color_per_row(self):
        # Arrange
        image = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2],
            [3, 3, 5, 5, 4, 4, 4],
            [3, 3, 3, 4, 4, 4, 4]], dtype=np.uint8)
        # Act
        actual = image_transition_color_per_row(image)
        # Assert
        expected = [
            [1],
            [1, 2],
            [1, 2],
            [3, 5, 4],
            [3, 4],
        ]
        self.assertEqual(actual, expected)

    def test_20000_image_transition_mass_per_row(self):
        # Arrange
        image = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2],
            [3, 3, 5, 5, 4, 4, 4],
            [3, 3, 3, 4, 4, 4, 4]], dtype=np.uint8)
        # Act
        actual = image_transition_mass_per_row(image)
        # Assert
        expected = [
            [7],
            [3, 4],
            [3, 4],
            [2, 2, 3],
            [3, 4],
        ]
        self.assertEqual(actual, expected)

    def test_20000_intersection_of_listlistint_identical(self):
        # Arrange
        items0 = [
            [7],
            [3, 4],
            [3, 4],
            [2, 2, 3],
            [3, 4],
        ]
        # Act
        actual = intersection_of_listlistint(items0, items0)
        # Assert
        expected = items0
        self.assertEqual(actual, expected)

    def test_20001_intersection_of_listlistint_some_overlap(self):
        # Arrange
        items0 = [
            [7],
            [3, 4],
            [3, 4],
            [2, 2, 3],
            [3, 4],
        ]
        items1 = [
            [7, 8],
            [3, 4],
            [3, 4],
            [2, 2, 3],
            [1, 2, 3, 4],
            [3, 4, 5, 6],
        ]
        # Act
        actual = intersection_of_listlistint(items0, items1)
        # Assert
        expected = [
            [3, 4],
            [3, 4],
            [2, 2, 3],
        ]
        self.assertEqual(actual, expected)

    def test_20002_intersection_of_listlistint_no_overlap(self):
        # Arrange
        items0 = [
            [7],
            [2, 2, 4],
            [9, 3],
        ]
        items1 = [
            [7, 8],
            [3, 4],
            [3, 4],
            [2, 2, 3],
            [1, 2, 3, 4],
            [3, 4, 5, 6],
        ]
        # Act
        actual = intersection_of_listlistint(items0, items1)
        # Assert
        expected = []
        self.assertEqual(actual, expected)

    def test_30000_image_transition_similarity_per_row_color_identical(self):
        # Arrange
        image0 = np.array([
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2],
            [3, 3, 5, 5, 4, 4, 4],
            [3, 3, 3, 4, 4, 4, 4]], dtype=np.uint8)
        # Act
        actual = image_transition_similarity_per_row(image0, image0, TransitionType.COLOR)
        # Assert
        expected = (5, 10)
        self.assertEqual(actual, expected)

    def test_30001_image_transition_similarity_per_row_color_some_overlap(self):
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
        actual = image_transition_similarity_per_row(image0, image1, TransitionType.COLOR)
        # Assert
        expected = (3, 10)
        self.assertEqual(actual, expected)
