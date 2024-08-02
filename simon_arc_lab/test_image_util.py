import unittest
import numpy as np
from .image_util import *

class TestImageUtil(unittest.TestCase):
    def test_image_create(self):
        actual = image_create(2, 3, 4)
        expected = np.array([
            [4, 4], 
            [4, 4], 
            [4, 4]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_rotate_cw(self):
        input = np.array([
            [3, 6],
            [2, 5],
            [1, 4]], dtype=np.uint8)
        expected = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        actual = image_rotate_cw(input)
        self.assertTrue(np.array_equal(actual, expected))

    def test_rotate_ccw(self):
        input = np.array([
            [4, 1],
            [5, 2],
            [6, 3]], dtype=np.uint8)
        expected = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        actual = image_rotate_ccw(input)
        self.assertTrue(np.array_equal(actual, expected))

    def test_rotate_180(self):
        input = np.array([
            [4, 1],
            [5, 2],
            [6, 3]], dtype=np.uint8)
        expected = np.array([
            [3, 6],
            [2, 5],
            [1, 4]], dtype=np.uint8)
        actual = image_rotate_180(input)
        self.assertTrue(np.array_equal(actual, expected))

    def test_flipx(self):
        input = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        expected = np.array([
            [3, 2, 1],
            [6, 5, 4]], dtype=np.uint8)
        actual = image_flipx(input)
        self.assertTrue(np.array_equal(actual, expected))

    def test_flipy(self):
        input = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        expected = np.array([
            [4, 5, 6],
            [1, 2, 3]], dtype=np.uint8)
        actual = image_flipy(input)
        self.assertTrue(np.array_equal(actual, expected))

    def test_flip_diagonal_a_square(self):
        input = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype=np.uint8)
        expected = np.array([
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]], dtype=np.uint8)
        actual = image_flip_diagonal_a(input)
        self.assertTrue(np.array_equal(actual, expected))

    def test_flip_diagonal_a_nonsquare(self):
        input = np.array([
            [0, 1, 2, 3],
            [0, 4, 5, 6],
            [0, 7, 8, 9]], dtype=np.uint8)
        expected = np.array([
            [0, 0, 0],
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]], dtype=np.uint8)
        actual = image_flip_diagonal_a(input)
        self.assertTrue(np.array_equal(actual, expected))

    def test_flip_diagonal_b_square(self):
        input = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]], dtype=np.uint8)
        expected = np.array([
            [9, 6, 3],
            [8, 5, 2],
            [7, 4, 1]], dtype=np.uint8)
        actual = image_flip_diagonal_b(input)
        self.assertTrue(np.array_equal(actual, expected))

    def test_flip_diagonal_b_nonsquare(self):
        input = np.array([
            [0, 1, 2, 3],
            [0, 4, 5, 6],
            [0, 7, 8, 9]], dtype=np.uint8)
        expected = np.array([
            [9, 6, 3],
            [8, 5, 2],
            [7, 4, 1],
            [0, 0, 0]], dtype=np.uint8)
        actual = image_flip_diagonal_b(input)
        self.assertTrue(np.array_equal(actual, expected))

    def test_image_translate_wrap_dxplus1(self):
        image = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]], dtype=np.uint8)
        actual = image_translate_wrap(image, 1, 0)
        expected = np.array([
            [3, 1, 2], 
            [6, 4, 5], 
            [9, 7, 8]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_image_translate_wrap_dxminus1(self):
        image = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]], dtype=np.uint8)
        actual = image_translate_wrap(image, -1, 0)
        expected = np.array([
            [2, 3, 1], 
            [5, 6, 4], 
            [8, 9, 7]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_image_translate_wrap_dyplus1(self):
        image = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]], dtype=np.uint8)
        actual = image_translate_wrap(image, 0, 1)
        expected = np.array([
            [7, 8, 9],
            [1, 2, 3], 
            [4, 5, 6]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_image_translate_wrap_dyminus1(self):
        image = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]], dtype=np.uint8)
        actual = image_translate_wrap(image, 0, -1)
        expected = np.array([
            [4, 5, 6], 
            [7, 8, 9],
            [1, 2, 3]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_image_replace_colors1(self):
        image = np.array([
            [4, 4, 5, 5, 6, 6], 
            [7, 7, 9, 9, 8, 8]], dtype=np.uint8)
        dict = {
            8: 9,
            9: 8,
        }
        actual = image_replace_colors(image, dict)
        expected = np.array([
            [4, 4, 5, 5, 6, 6], 
            [7, 7, 8, 8, 9, 9]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_image_replace_colors2(self):
        image = np.array([
            [1, 2, 3], 
            [3, 2, 1]], dtype=np.uint8)
        dict = {
            1: 2,
            2: 3,
            3: 1
        }
        actual = image_replace_colors(image, dict)
        expected = np.array([
            [2, 3, 1], 
            [1, 3, 2]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_image_get_row_as_list(self):
        image = np.array([
            [1, 2], 
            [3, 4], 
            [5, 6], 
            [7, 8]], dtype=np.uint8)

        with self.assertRaises(ValueError):
            image_get_row_as_list(image, -1)
        with self.assertRaises(ValueError):
            image_get_row_as_list(image, 4)

        self.assertEqual(image_get_row_as_list(image, 0), [1, 2])
        self.assertEqual(image_get_row_as_list(image, 1), [3, 4])
        self.assertEqual(image_get_row_as_list(image, 2), [5, 6])
        self.assertEqual(image_get_row_as_list(image, 3), [7, 8])

    def test_image_get_column_as_list(self):
        image = np.array([
            [1, 2, 3, 4], 
            [5, 6, 7, 8]], dtype=np.uint8)

        with self.assertRaises(ValueError):
            image_get_column_as_list(image, -1)
        with self.assertRaises(ValueError):
            image_get_column_as_list(image, 4)

        self.assertEqual(image_get_column_as_list(image, 0), [1, 5])
        self.assertEqual(image_get_column_as_list(image, 1), [2, 6])
        self.assertEqual(image_get_column_as_list(image, 2), [3, 7])
        self.assertEqual(image_get_column_as_list(image, 3), [4, 8])

if __name__ == '__main__':
    unittest.main()
