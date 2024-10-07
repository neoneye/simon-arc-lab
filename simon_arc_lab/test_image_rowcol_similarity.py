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

    # def test_20000_image_rowcol_similarity_same_size(self):
    #     # Arrange
    #     image0 = np.array([
    #         [1, 1, 1, 2, 2, 2, 2],
    #         [1, 1, 1, 2, 2, 2, 2],
    #         [1, 1, 1, 2, 2, 2, 2],
    #         [3, 3, 5, 5, 4, 4, 4],
    #         [3, 3, 3, 4, 4, 4, 4]], dtype=np.uint8)
    #     image1 = np.array([
    #         [1, 1, 1, 2, 2, 2, 2],
    #         [1, 1, 1, 2, 2, 2, 2],
    #         [1, 1, 5, 5, 2, 2, 2],
    #         [3, 3, 3, 4, 4, 4, 4],
    #         [3, 3, 3, 4, 4, 4, 4]], dtype=np.uint8)
    #     # Act
    #     actual = image_rowcol_similarity(image0, image1)
    #     print(actual)
    #     # Assert
    #     # self.assertEqual(actual, expected)
