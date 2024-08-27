import unittest
import numpy as np
from .image_bresenham_line import *
from .image_util import *

class TestImageBresenhamLine(unittest.TestCase):
    def test_bresenham_line_diagonal(self):
        image = np.zeros((3, 3), dtype=np.uint8)
        actual = image_bresenham_line(image, 0, 0, 2, 2, 1)
        expected = np.zeros((3, 3), dtype=np.uint8)
        expected[0, 0] = 1
        expected[1, 1] = 1
        expected[2, 2] = 1
        np.testing.assert_array_equal(actual, expected)

    def test_bresenham_line_horizontal(self):
        image = np.zeros((3, 4), dtype=np.uint8)
        actual = image_bresenham_line(image, 0, 0, 3, 0, 1)
        expected = np.zeros((3, 4), dtype=np.uint8)
        expected[0, 0] = 1
        expected[0, 1] = 1
        expected[0, 2] = 1
        expected[0, 3] = 1
        np.testing.assert_array_equal(actual, expected)

    def test_bresenham_line_vertical(self):
        image = np.zeros((3, 4), dtype=np.uint8)
        actual = image_bresenham_line(image, 1, 0, 1, 2, 1)
        expected = np.zeros((3, 4), dtype=np.uint8)
        expected[0, 1] = 1
        expected[1, 1] = 1
        expected[2, 1] = 1
        np.testing.assert_array_equal(actual, expected)

    def test_bresenham_line_outside_topbottom(self):
        image = np.zeros((4, 5), dtype=np.uint8)
        actual = image_bresenham_line(image, 1, -1, 3, 4, 1)
        expected = np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_bresenham_line_outside_leftright(self):
        image = np.zeros((5, 4), dtype=np.uint8)
        actual = image_bresenham_line(image, -1, 1, 4, 3, 1)
        expected = np.array([
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

if __name__ == '__main__':
    unittest.main()
