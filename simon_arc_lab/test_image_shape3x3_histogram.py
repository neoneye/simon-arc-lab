import unittest
import numpy as np
from .image_shape3x3_histogram import *

class TestImageShape3x3Histogram(unittest.TestCase):
    def test_all9_unique_colors(self):
        image = np.array([
            [1, 2, 3, 1], 
            [4, 5, 6, 4],
            [7, 8, 9, 7]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_all9(image)
        expected = np.array([
            [4, 6, 6, 4], 
            [6, 9, 9, 6], 
            [4, 6, 6, 4]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_all9_same_color(self):
        image = np.array([
            [5, 5, 5, 5], 
            [5, 5, 5, 5], 
            [5, 5, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_all9(image)
        expected = np.array([
            [1, 1, 1, 1], 
            [1, 1, 1, 1], 
            [1, 1, 1, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_all9_two_colors(self):
        image = np.array([
            [6, 6, 5, 5], 
            [6, 6, 5, 5], 
            [6, 6, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_all9(image)
        expected = np.array([
            [1, 2, 2, 1], 
            [1, 2, 2, 1], 
            [1, 2, 2, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_aroundcenter_unique_colors(self):
        image = np.array([
            [1, 2, 3, 1], 
            [4, 5, 6, 4],
            [7, 8, 9, 7]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_around_center(image)
        expected = np.array([
            [3, 5, 5, 3], 
            [5, 8, 8, 5], 
            [3, 5, 5, 3]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_aroundcenter_same_color(self):
        image = np.array([
            [5, 5, 5, 5], 
            [5, 5, 5, 5], 
            [5, 5, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_around_center(image)
        expected = np.array([
            [1, 1, 1, 1], 
            [1, 1, 1, 1], 
            [1, 1, 1, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_aroundcenter_two_colors(self):
        image = np.array([
            [6, 6, 5, 5], 
            [6, 6, 5, 5], 
            [6, 6, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_around_center(image)
        expected = np.array([
            [1, 2, 2, 1], 
            [1, 2, 2, 1], 
            [1, 2, 2, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_aroundcenter_one_pixel(self):
        image = np.array([
            [6, 6, 6, 6], 
            [6, 9, 6, 6], 
            [6, 6, 6, 6]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_around_center(image)
        expected = np.array([
            [2, 2, 2, 1], 
            [2, 1, 2, 1], 
            [2, 2, 2, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_corners_unique_colors(self):
        image = np.array([
            [1, 2, 3, 1], 
            [4, 5, 6, 4],
            [7, 8, 9, 7]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_in_corners(image)
        expected = np.array([
            [1, 2, 2, 1], 
            [2, 4, 4, 2], 
            [1, 2, 2, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_corners_same_color(self):
        image = np.array([
            [5, 5, 5, 5], 
            [5, 5, 5, 5], 
            [5, 5, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_in_corners(image)
        expected = np.array([
            [1, 1, 1, 1], 
            [1, 1, 1, 1], 
            [1, 1, 1, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_corners_two_colors(self):
        image = np.array([
            [6, 6, 5, 5], 
            [6, 6, 5, 5], 
            [6, 6, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_in_corners(image)
        expected = np.array([
            [1, 2, 2, 1], 
            [1, 2, 2, 1], 
            [1, 2, 2, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_corners_around_plus_a(self):
        image = np.array([
            [6, 6, 5, 6], 
            [5, 5, 5, 5], 
            [6, 6, 5, 6]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_in_corners(image)
        expected = np.array([
            [1, 1, 1, 1], 
            [1, 2, 1, 1], 
            [1, 1, 1, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_corners_around_plus_a(self):
        image = np.array([
            [5, 1, 5, 2], 
            [5, 5, 5, 5], 
            [5, 4, 5, 3]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_in_corners(image)
        expected = np.array([
            [1, 1, 1, 1], 
            [2, 1, 4, 1], 
            [1, 1, 1, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_diamond4_unique_colors(self):
        image = np.array([
            [1, 2, 3, 1], 
            [4, 5, 6, 4],
            [7, 8, 9, 7]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_in_diamond4(image)
        expected = np.array([
            [2, 3, 3, 2], 
            [3, 4, 4, 3], 
            [2, 3, 3, 2]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_diamond4_same_color(self):
        image = np.array([
            [5, 5, 5, 5], 
            [5, 5, 5, 5], 
            [5, 5, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_in_diamond4(image)
        expected = np.array([
            [1, 1, 1, 1], 
            [1, 1, 1, 1], 
            [1, 1, 1, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_diamond4_two_colors(self):
        image = np.array([
            [6, 6, 5, 5], 
            [6, 6, 5, 5], 
            [6, 6, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_in_diamond4(image)
        expected = np.array([
            [1, 2, 2, 1], 
            [1, 2, 2, 1], 
            [1, 2, 2, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_diamond4_a_diamond_of_one_color_a(self):
        image = np.array([
            [5, 5, 6, 5, 5], 
            [5, 6, 6, 6, 5], 
            [5, 5, 6, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_in_diamond4(image)
        expected = np.array([
            [1, 2, 2, 2, 1], 
            [2, 2, 1, 2, 2], 
            [1, 2, 2, 2, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_diamond4_a_diamond_of_one_color_b(self):
        image = np.array([
            [5, 5, 6, 5, 5], 
            [5, 6, 5, 6, 5], 
            [5, 5, 6, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_in_diamond4(image)
        expected = np.array([
            [1, 2, 1, 2, 1], 
            [2, 1, 1, 1, 2], 
            [1, 2, 1, 2, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_diamond5_unique_colors(self):
        image = np.array([
            [1, 2, 3, 1], 
            [4, 5, 6, 4],
            [7, 8, 9, 7]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_in_diamond5(image)
        expected = np.array([
            [3, 4, 4, 3], 
            [4, 5, 5, 4], 
            [3, 4, 4, 3]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_diamond5_same_color(self):
        image = np.array([
            [5, 5, 5, 5], 
            [5, 5, 5, 5], 
            [5, 5, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_in_diamond5(image)
        expected = np.array([
            [1, 1, 1, 1], 
            [1, 1, 1, 1], 
            [1, 1, 1, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_diamond5_two_colors(self):
        image = np.array([
            [6, 6, 5, 5], 
            [6, 6, 5, 5], 
            [6, 6, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_in_diamond5(image)
        expected = np.array([
            [1, 2, 2, 1], 
            [1, 2, 2, 1], 
            [1, 2, 2, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_diamond5_a_diamond_of_one_color(self):
        image = np.array([
            [5, 5, 6, 5, 5], 
            [5, 6, 6, 6, 5], 
            [5, 5, 6, 5, 5]], dtype=np.uint8)
        actual = ImageShape3x3Histogram.number_of_unique_colors_in_diamond5(image)
        expected = np.array([
            [1, 2, 2, 2, 1], 
            [2, 2, 1, 2, 2], 
            [1, 2, 2, 2, 1]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
