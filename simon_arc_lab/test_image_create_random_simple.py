import unittest
import numpy as np
from .image_util import *
from .image_create_random_simple import *

class TestImageCreateRandomSimple(unittest.TestCase):
    def test_image_create_random_with_two_colors_50(self):
        actual = image_create_random_with_two_colors(2, 3, 4, 5, 0.5, 0)

        expected = np.zeros((3, 2), dtype=np.uint8)
        expected[0:3, 0:2] = [
            [4, 5],
            [5, 4],
            [5, 4]]

        self.assertTrue(np.array_equal(actual, expected))

    def test_image_create_random_with_two_colors_25(self):
        actual = image_create_random_with_two_colors(2, 4, 8, 9, 0.25, 0)

        expected = np.zeros((4, 2), dtype=np.uint8)
        expected[0:4, 0:2] = [
            [8, 9],
            [8, 8],
            [9, 8],
            [8, 8]]

        self.assertTrue(np.array_equal(actual, expected))

    def test_image_create_random_with_three_colors_1_1_1(self):
        actual = image_create_random_with_three_colors(2, 3, 4, 5, 6, 1, 1, 1, 0)

        expected = np.zeros((3, 2), dtype=np.uint8)
        expected[0:3, 0:2] = [
            [6, 6],
            [5, 4],
            [5, 4]]

        self.assertTrue(np.array_equal(actual, expected))

    def test_image_create_random_with_three_colors_3_2_1(self):
        actual = image_create_random_with_three_colors(2, 3, 4, 5, 6, 3, 2, 1, 0)

        expected = np.zeros((3, 2), dtype=np.uint8)
        expected[0:3, 0:2] = [
            [4, 6],
            [5, 4],
            [5, 4]]

        self.assertTrue(np.array_equal(actual, expected))

    def test_image_create_random_with_four_colors_1_1_1_1(self):
        actual = image_create_random_with_four_colors(2, 4, 4, 5, 6, 7, 1, 1, 1, 1, 0)

        expected = np.zeros((4, 2), dtype=np.uint8)
        expected[0:4, 0:2] = [
            [7, 5],
            [6, 7],
            [5, 6],
            [4, 4]]

        self.assertTrue(np.array_equal(actual, expected))

    def test_image_create_random_with_four_colors_3_3_1_1(self):
        actual = image_create_random_with_four_colors(2, 4, 4, 5, 6, 7, 3, 3, 1, 1, 0)

        expected = np.zeros((4, 2), dtype=np.uint8)
        expected[0:4, 0:2] = [
            [7, 5],
            [6, 4],
            [5, 5],
            [4, 4]]

        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
