import unittest
import numpy as np
from .image_bigram import *

class TestImageBigram(unittest.TestCase):
    def test_10000_extract_bigrams(self):
        # Arrange
        pixels = [1, 2, 3, 3, 3, 1, 1, 5, 5]
        # Act
        actual = extract_bigrams(pixels, 254)
        # Assert
        expected = [
            (254, 1), (1, 2), (2, 3), (3, 3), (3, 3), (3, 1), (1, 1), (1, 5), (5, 5), (5, 254)
        ]
        self.assertEqual(actual, expected)

    def test_20000_image_bigrams_from_left_to_right(self):
        # Arrange
        input = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        # Act
        actual = image_bigrams_from_top_to_bottom(input, 255)
        # Assert
        expected = [
            (255, 1), (1, 2), (2, 3), (3, 255), (255, 4), (4, 5), (5, 6), (6, 255)
        ]
        self.assertEqual(actual, expected)

    def test_30000_image_bigrams_from_left_to_right(self):
        # Arrange
        input = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        # Act
        actual = image_bigrams_from_left_to_right(input, 255)
        # Assert
        expected = [
            (255, 1), (1, 4), (4, 255), (255, 2), (2, 5), (5, 255), (255, 3), (3, 6), (6, 255)
        ]
        self.assertEqual(actual, expected)

    def test_40000_image_bigrams_from_topleft_to_bottomright(self):
        # Arrange
        input = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        # Act
        actual = image_bigrams_from_topleft_to_bottomright(input, 255)

        # Assert
        expected = [
            (255, 3), (3, 255), (255, 2), (2, 6), (6, 255), (255, 1), (1, 5), (5, 255), (255, 4), (4, 255)
        ]
        self.assertEqual(actual, expected)

    def test_50000_image_bigrams_from_topright_to_bottomleft(self):
        # Arrange
        input = np.array([
            [1, 4],
            [2, 5],
            [3, 6]], dtype=np.uint8)
        # Act
        actual = image_bigrams_from_topright_to_bottomleft(input, 255)

        # Assert
        expected = [
            (255, 6), (6, 255), (255, 5), (5, 3), (3, 255), (255, 4), (4, 2), (2, 255), (255, 1), (1, 255)
        ]
        self.assertEqual(actual, expected)

    def test_60000_sorted_unique_bigrams_a(self):
        # Arrange
        input = [(9, 1), (1, 1), (1, 2), (2, 2), (2, 2)]
        # Act
        actual = sorted_unique_bigrams(input)
        # Assert
        expected = [(1, 2), (1, 9)]
        self.assertEqual(actual, expected)

    def test_70000_image_bigrams_direction_all_1x1(self):
        # Arrange
        input = np.array([[1]], dtype=np.uint8)
        # Act
        actual = image_bigrams_direction_all(input, 255)

        # Assert
        expected = [
            (1, 255)
        ]
        self.assertEqual(actual, expected)

    def test_70001_image_bigrams_direction_all_2x2(self):
        # Arrange
        input = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        # Act
        actual = image_bigrams_direction_all(input, 255)
        # Assert
        expected = [
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 255),
            (2, 3),
            (2, 4),
            (2, 255),
            (3, 4),
            (3, 255),
            (4, 255)
        ]
        self.assertEqual(actual, expected)

    def test_80000_image_bigrams_direction_leftright(self):
        # Arrange
        input = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        # Act
        actual = image_bigrams_direction_leftright(input, 255)
        # Assert
        expected = [
            (1, 2),
            (1, 255),
            (2, 255),
            (3, 4),
            (3, 255),
            (4, 255)
        ]
        self.assertEqual(actual, expected)

    def test_90000_image_bigrams_direction_topbottom(self):
        # Arrange
        input = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        # Act
        actual = image_bigrams_direction_topbottom(input, 255)
        # Assert
        expected = [
            (1, 3),
            (1, 255),
            (2, 4),
            (2, 255),
            (3, 255),
            (4, 255)
        ]
        self.assertEqual(actual, expected)

    def test_100000_image_bigrams_direction_topleftbottomright(self):
        # Arrange
        input = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        # Act
        actual = image_bigrams_direction_topleftbottomright(input, 255)
        # Assert
        expected = [
            (1, 4),
            (1, 255),
            (2, 255),
            (3, 255),
            (4, 255)
        ]
        self.assertEqual(actual, expected)

    def test_110000_image_bigrams_direction_toprightbottomleft(self):
        # Arrange
        input = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        # Act
        actual = image_bigrams_direction_toprightbottomleft(input, 255)
        # Assert
        expected = [
            (1, 255),
            (2, 3),
            (2, 255),
            (3, 255),
            (4, 255)
        ]
        self.assertEqual(actual, expected)
