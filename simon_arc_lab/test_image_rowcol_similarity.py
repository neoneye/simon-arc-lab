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

    def test_20000_intersectionset_of_listlistint_identical(self):
        # Arrange
        items0 = [
            [7],
            [3, 4],
            [3, 4],
            [2, 2, 3],
            [3, 4],
        ]
        # Act
        actual = intersectionset_of_listlistint(items0, items0)
        # Assert
        expected = set(map(tuple, items0))
        self.assertEqual(actual, expected)

    def test_20001_intersectionset_of_listlistint_with_some_overlap(self):
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
        actual = intersectionset_of_listlistint(items0, items1)
        # Assert
        expected = set(map(tuple, [
            [3, 4],
            [2, 2, 3],
        ]))
        self.assertEqual(actual, expected)

    def test_20002_intersectionset_of_listlistint_no_overlap(self):
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
        actual = intersectionset_of_listlistint(items0, items1)
        # Assert
        expected = set()
        self.assertEqual(actual, expected)

    def test_30000_unionset_of_listlistint_identical(self):
        # Arrange
        items0 = [
            [7],
            [2, 2, 4],
            [9, 3],
        ]
        # Act
        actual = unionset_of_listlistint(items0, items0)
        # Assert
        expected = set(map(tuple, [
            [2, 2, 4],
            [7],
            [9, 3],
        ]))
        self.assertEqual(actual, expected)

    def test_30001_unionset_of_listlistint_some_overlap(self):
        # Arrange
        items0 = [
            [7],
            [2, 2, 4],
            [4, 3, 1],
            [9, 3],
        ]
        items1 = [
            [7],
            [8, 8, 3],
            [2, 2, 4],
            [9, 3],
        ]
        # Act
        actual = unionset_of_listlistint(items0, items1)
        # Assert
        expected = set(map(tuple, [
            [7],
            [9, 3],
            [2, 2, 4],
            [4, 3, 1],
            [8, 8, 3],
        ]))
        self.assertEqual(actual, expected)

    def test_30002_unionset_of_listlistint_no_overlap(self):
        # Arrange
        items0 = [
            [7],
            [1, 2, 3],
        ]
        items1 = [
            [5, 5],
            [4, 5, 6],
        ]
        # Act
        actual = unionset_of_listlistint(items0, items1)
        # Assert
        expected = set(map(tuple, [
            [7],
            [1, 2, 3],
            [5, 5],
            [4, 5, 6],
        ]))
        self.assertEqual(actual, expected)

    def test_40000_image_transition_similarity_per_row_color_identical(self):
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
        expected = (3, 3)
        self.assertEqual(actual, expected)

    def test_40001_image_transition_similarity_per_row_color_some_overlap(self):
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
        expected = (2, 4)
        self.assertEqual(actual, expected)

    def test_40002_image_transition_similarity_per_row_color_no_overlap(self):
        # Arrange
        image0 = np.array([
            [1, 1, 1, 2, 2, 2, 7],
            [1, 1, 1, 2, 2, 2, 7],
            [1, 1, 1, 2, 2, 2, 7],
            [3, 3, 5, 5, 4, 4, 4],
            [3, 3, 3, 4, 4, 4, 7]], dtype=np.uint8)
        image1 = np.array([
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 5, 5, 2, 2, 2],
            [3, 3, 3, 4, 4, 4, 4],
            [3, 3, 3, 4, 4, 4, 4]], dtype=np.uint8)
        # Act
        actual = image_transition_similarity_per_row(image0, image1, TransitionType.COLOR)
        # Assert
        expected = (0, 6)
        self.assertEqual(actual, expected)

    def test_50000_image_transition_similarity_per_column_color_identical(self):
        # Arrange
        image0 = np.array([
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2],
            [3, 3, 5, 5, 4, 4, 4],
            [3, 3, 3, 4, 4, 4, 4]], dtype=np.uint8)
        # Act
        actual = image_transition_similarity_per_column(image0, image0, TransitionType.COLOR)
        # Assert
        expected = (4, 4)
        self.assertEqual(actual, expected)

    def test_50001_image_transition_similarity_per_column_color_identical(self):
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
        actual = image_transition_similarity_per_column(image0, image1, TransitionType.COLOR)
        # Assert
        expected = (4, 4)
        self.assertEqual(actual, expected)

    def test_50002_image_transition_similarity_per_column_color_some_overlap(self):
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
            [1, 1, 1, 2, 2, 2, 2],
            [3, 3, 7, 7, 4, 4, 4],
            [3, 3, 3, 4, 4, 4, 4]], dtype=np.uint8)
        # Act
        actual = image_transition_similarity_per_column(image0, image1, TransitionType.COLOR)
        # Assert
        expected = (2, 6)
        self.assertEqual(actual, expected)

    def test_50003_image_transition_similarity_per_column_color_no_overlap(self):
        # Arrange
        image0 = np.array([
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2],
            [3, 3, 5, 5, 4, 4, 4],
            [3, 3, 3, 4, 4, 4, 4]], dtype=np.uint8)
        image1 = np.array([
            [9, 9, 9, 9, 9, 9, 9],
            [1, 1, 1, 2, 2, 2, 2],
            [1, 1, 1, 2, 2, 2, 2],
            [3, 3, 7, 7, 4, 4, 4],
            [3, 3, 3, 4, 4, 4, 4]], dtype=np.uint8)
        # Act
        actual = image_transition_similarity_per_column(image0, image1, TransitionType.COLOR)
        # Assert
        expected = (0, 8)
        self.assertEqual(actual, expected)

