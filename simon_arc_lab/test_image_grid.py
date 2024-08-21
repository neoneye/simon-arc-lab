import unittest
import numpy as np
from .image_grid import ImageGridBuilder

class TestImageGrid(unittest.TestCase):
    def test_10000_default_constructor(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        height, width = image.shape
        # Act
        builder = ImageGridBuilder(width, height)
        actual = builder.draw(image, 0)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 2, 0, 3, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 4, 0, 5, 0, 6, 0],
            [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10001_set_size2(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        height, width = image.shape
        # Act
        builder = ImageGridBuilder(width, height)
        builder.set_cell_size(2)
        actual = builder.draw(image, 0)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 2, 2, 0, 3, 3, 0],
            [0, 1, 1, 0, 2, 2, 0, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 4, 4, 0, 5, 5, 0, 6, 6, 0],
            [0, 4, 4, 0, 5, 5, 0, 6, 6, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_10002_set_cell_size_random(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        height, width = image.shape
        # Act
        builder = ImageGridBuilder(width, height)
        builder.set_cell_size_random(43, 2, 5)
        self.assertEqual(builder.cell_widths, [2, 5, 4])
        self.assertEqual(builder.cell_heights, [2, 4])
        actual = builder.draw(image, 0)
        height, width = actual.shape
        # # Assert
        self.assertEqual(width, 15) # 1 + 2 + 1 + 5 + 1 + 4 + 1
        self.assertEqual(height, 9) # 1 + 2 + 1 + 4 + 1

    def test_10003_set_custom_separator_size(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        height, width = image.shape
        # Act
        builder = ImageGridBuilder(width, height)
        builder.set_left_separator_size(0)
        builder.set_right_separator_size(1)
        builder.set_top_separator_size(2)
        builder.set_bottom_separator_size(3)
        actual = builder.draw(image, 0)
        # Assert
        expected = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 2, 0, 3, 0],
            [0, 0, 0, 0, 0, 0],
            [4, 0, 5, 0, 6, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

