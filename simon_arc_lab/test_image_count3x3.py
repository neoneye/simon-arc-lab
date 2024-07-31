import unittest
import numpy as np
from .image_count3x3 import *

class TestImageCount3x3(unittest.TestCase):
    def test_count_same_color_as_center_with_one_neighbor_nowrap_1(self):
        image = np.array([[1, 5, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
        actual = count_same_color_as_center_with_one_neighbor_nowrap(image, 0, -1)
        expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_count_same_color_as_center_with_one_neighbor_nowrap_2(self):
        image = np.array([[5, 2, 3], [4, 5, 6], [7, 8, 5]], dtype=np.uint8)
        actual = count_same_color_as_center_with_one_neighbor_nowrap(image, -1, -1)
        expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_count_neighbors_with_same_color_nowrap_1(self):
        image = np.array([[9, 7, 9], [7, 7, 7], [9, 7, 9]], dtype=np.uint8)
        actual = count_neighbors_with_same_color_nowrap(image)
        expected = np.array([[0, 3, 0], [3, 4, 3], [0, 3, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_count_neighbors_with_same_color_nowrap_2(self):
        image = np.array([[6, 8, 8], [8, 6, 8], [8, 8, 6]], dtype=np.uint8)
        actual = count_neighbors_with_same_color_nowrap(image)
        expected = np.array([[1, 3, 2], [3, 2, 3], [2, 3, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_all_neighbors_matching_center_nowrap_1(self):
        image = np.array([[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]], dtype=np.uint8)
        actual = all_neighbors_matching_center_nowrap(image)
        expected = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_pixels_with_k_matching_neighbors_nowrap_1(self):
        image = np.array([[9, 7, 9], [7, 7, 7], [9, 7, 9]], dtype=np.uint8)
        actual = pixels_with_k_matching_neighbors_nowrap(image, 3)
        expected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_pixels_with_k_matching_neighbors_nowrap_2(self):
        image = np.array([[3, 7, 3, 3], [3, 3, 7, 3], [3, 3, 3, 7]], dtype=np.uint8)
        actual = pixels_with_k_matching_neighbors_nowrap(image, 1)
        expected = np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
