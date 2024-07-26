import unittest
import numpy as np
from .image_shape3x3_histogram import *

class TestImageShape3x3Histogram(unittest.TestCase):
    def test_all_unique_colors(self):
        image = np.array([
            [1, 2, 3, 1], 
            [4, 5, 6, 4],
            [7, 8, 9, 7]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors(image)
        expected = np.array([
            [4, 6, 6, 4], 
            [6, 9, 9, 6], 
            [4, 6, 6, 4]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_same_color(self):
        image = np.array([
            [5, 5, 5, 5], 
            [5, 5, 5, 5], 
            [5, 5, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors(image)
        expected = np.array([
            [1, 1, 1, 1], 
            [1, 1, 1, 1], 
            [1, 1, 1, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_two_colors(self):
        image = np.array([
            [6, 6, 5, 5], 
            [6, 6, 5, 5], 
            [6, 6, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors(image)
        expected = np.array([
            [1, 2, 2, 1], 
            [1, 2, 2, 1], 
            [1, 2, 2, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
