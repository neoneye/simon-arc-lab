import unittest
import numpy as np
from .image_shape3x3_center import *

class TestImageShape3x3Center(unittest.TestCase):
    def test_10000_no_neighbors_with_same_color(self):
        image = np.array([
            [1, 2, 3, 4], 
            [5, 6, 7, 8],
            [9, 0, 1, 2]], dtype=np.uint8)
        actual = ImageShape3x3Center.apply(image)
        expected = np.array([
            [0, 0, 0, 0], 
            [0, 0, 0, 0], 
            [0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_10001_vertical_lines(self):
        image = np.array([
            [1, 2, 3, 4], 
            [1, 2, 3, 4], 
            [1, 2, 3, 4]], dtype=np.uint8)
        actual = ImageShape3x3Center.apply(image)
        a = ImageShape3x3Center.BOTTOM # same color below
        b = ImageShape3x3Center.TOP | ImageShape3x3Center.BOTTOM # same color above and below
        c = ImageShape3x3Center.TOP # same color above
        self.assertEqual(b, 0x42)
        expected = np.array([
            [a, a, a, a], 
            [b, b, b, b], 
            [c, c, c, c]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_10002_horizontal_lines(self):
        image = np.array([
            [1, 1, 1, 1], 
            [2, 2, 2, 2], 
            [3, 3, 3, 3]], dtype=np.uint8)
        actual = ImageShape3x3Center.apply(image)
        a = ImageShape3x3Center.RIGHT # same color to the right
        b = ImageShape3x3Center.RIGHT | ImageShape3x3Center.LEFT # same color to the right and left
        c = ImageShape3x3Center.LEFT # same color to the left
        self.assertEqual(b, 0x18)
        expected = np.array([
            [a, b, b, c], 
            [a, b, b, c], 
            [a, b, b, c]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_10003_diagonal_lines_topleft_bottomright(self):
        image = np.array([
            [1, 3, 2, 1], 
            [2, 1, 3, 2], 
            [3, 2, 1, 3]], dtype=np.uint8)
        actual = ImageShape3x3Center.apply(image)
        a = ImageShape3x3Center.BOTTOM_RIGHT
        b = ImageShape3x3Center.TOP_LEFT | ImageShape3x3Center.BOTTOM_RIGHT
        c = ImageShape3x3Center.TOP_LEFT
        self.assertEqual(b, 0x81)
        expected = np.array([
            [a, a, a, 0], 
            [a, b, b, c], 
            [0, c, c, c]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_10004_diagonal_lines_topright_bottomleft(self):
        image = np.array([
            [3, 2, 1, 3],
            [2, 1, 3, 2], 
            [1, 3, 2, 1]], dtype=np.uint8)
        actual = ImageShape3x3Center.apply(image)
        a = ImageShape3x3Center.BOTTOM_LEFT
        b = ImageShape3x3Center.TOP_RIGHT | ImageShape3x3Center.BOTTOM_LEFT
        c = ImageShape3x3Center.TOP_RIGHT
        self.assertEqual(b, 0x24)
        expected = np.array([
            [0, a, a, a],
            [c, b, b, a], 
            [c, c, c, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_10005_all_same_color(self):
        image = np.array([
            [9, 9, 9, 9], 
            [9, 9, 9, 9], 
            [9, 9, 9, 9]], dtype=np.uint8)
        actual = ImageShape3x3Center.apply(image)
        expected = np.array([
            [0xd0, 0xf8, 0xf8, 0x68], 
            [0xd6, 0xff, 0xff, 0x6b], 
            [0x16, 0x1f, 0x1f, 0x0b]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_20000_shape_id_list(self):
        image = np.array([
            [1, 2, 3, 4], 
            [5, 6, 7, 8],
            [9, 0, 1, 2]], dtype=np.uint8)
        actual = ImageShape3x3Center.shape_id_list(image)
        expected = [0]
        self.assertEqual(actual, expected)

    def test_20001_shape_id_list(self):
        image = np.array([
            [1, 2, 3, 4], 
            [1, 2, 3, 4], 
            [1, 2, 3, 4]], dtype=np.uint8)
        actual = ImageShape3x3Center.shape_id_list(image)
        a = ImageShape3x3Center.BOTTOM # same color below
        b = ImageShape3x3Center.TOP | ImageShape3x3Center.BOTTOM # same color above and below
        c = ImageShape3x3Center.TOP # same color above
        expected = [c, a, b]
        self.assertEqual(actual, expected)

    def test_20002_shape_id_list(self):
        image = np.array([
            [1, 3, 2, 1], 
            [2, 1, 3, 2], 
            [3, 2, 1, 3]], dtype=np.uint8)
        actual = ImageShape3x3Center.shape_id_list(image)
        a = ImageShape3x3Center.BOTTOM_RIGHT
        b = ImageShape3x3Center.TOP_LEFT | ImageShape3x3Center.BOTTOM_RIGHT
        c = ImageShape3x3Center.TOP_LEFT
        expected = [0, c, a, b]
        self.assertEqual(actual, expected)

    def test_20003_shape_id_list(self):
        image = np.array([
            [3, 2, 1, 3],
            [2, 1, 3, 2], 
            [1, 3, 2, 1]], dtype=np.uint8)
        actual = ImageShape3x3Center.shape_id_list(image)
        a = ImageShape3x3Center.BOTTOM_LEFT
        b = ImageShape3x3Center.TOP_RIGHT | ImageShape3x3Center.BOTTOM_LEFT
        c = ImageShape3x3Center.TOP_RIGHT
        expected = [0, c, a, b]
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
