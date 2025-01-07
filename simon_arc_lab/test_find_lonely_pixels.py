import unittest
import numpy as np
from .find_lonely_pixels import *

class TestFindLonelyPixels(unittest.TestCase):
    def test_10000_one_isolated_pixel(self):
        # Arrange
        image = np.array([
            [3, 3, 3, 3, 3],
            [3, 3, 7, 3, 3],
            [3, 3, 3, 3, 3]], dtype=np.uint8)
        # Act
        actual = find_lonely_pixels(image)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10001_two_isolated_pixels(self):
        # Arrange
        image = np.array([
            [3, 3, 7, 3, 3],
            [3, 3, 3, 3, 3],
            [3, 3, 6, 3, 3]], dtype=np.uint8)
        # Act
        actual = find_lonely_pixels(image)
        # Assert
        expected = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10002_all_different_colors(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype=np.uint8)
        # Act
        actual = find_lonely_pixels(image)
        # Assert
        expected = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10003_checkerboard(self):
        # Arrange
        image = np.array([
            [8, 8, 8, 8],
            [8, 1, 2, 8],
            [8, 2, 1, 8],
            [8, 1, 2, 8],
            [8, 8, 8, 8]], dtype=np.uint8)
        # Act
        actual = find_lonely_pixels(image)
        # Assert
        expected = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10004_alternating_along_diagonal_line(self):
        # Arrange
        image = np.array([
            [5, 3, 3, 3, 3],
            [3, 6, 3, 3, 3],
            [3, 3, 5, 3, 3],
            [3, 3, 3, 6, 3],
            [3, 3, 5, 3, 3],
            [3, 6, 3, 3, 3]], dtype=np.uint8)
        # Act
        actual = find_lonely_pixels(image)
        # Assert
        expected = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10005_alternating_snake_in_maze(self):
        # Arrange
        image = np.array([
            [3, 3, 5, 3, 3],
            [3, 3, 7, 3, 3],
            [3, 3, 5, 7, 5],
            [3, 3, 3, 3, 7],
            [3, 7, 5, 7, 5],
            [3, 5, 3, 3, 3],
            [3, 7, 3, 3, 3]], dtype=np.uint8)
        # Act
        actual = find_lonely_pixels(image)
        # Assert
        expected = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_20000_no_alternating_spiral(self):
        # Arrange
        image = np.array([
            [3, 3, 3, 3, 3],
            [8, 8, 8, 8, 3],
            [3, 3, 3, 8, 3],
            [3, 8, 8, 8, 3],
            [3, 3, 3, 3, 3]], dtype=np.uint8)
        # Act
        actual = find_lonely_pixels(image)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    # Diagonal lines are recognized as isolated pixels.
    # I don't want this behavior. Only isolated pixels should be recognized.
    # However when there are a snake of alternating colors, then there will be
    # two same colored pixels that are diagonally adjacent.
    # def xtest_20001_not_alternating_along_solid_diagonal(self):
    #     # Arrange
    #     image = np.array([
    #         [5, 3, 3, 3, 3],
    #         [3, 5, 3, 3, 3],
    #         [3, 3, 5, 3, 3],
    #         [3, 3, 3, 5, 3],
    #         [3, 3, 5, 3, 3],
    #         [3, 5, 3, 3, 3]], dtype=np.uint8)
    #     # Act
    #     actual = find_lonely_pixels(image)
    #     # Assert
    #     expected = np.array([
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0]], dtype=np.uint8)
    #     np.testing.assert_array_equal(actual, expected)

