import unittest
import numpy as np
from .image_shape2x2 import *

class TestImageShape2x2(unittest.TestCase):
    def test_no_neighbors_with_same_color(self):
        image = np.array([
            [1, 2, 3], 
            [5, 6, 7]], dtype=np.uint8)
        actual = ImageShape2x2.apply(image)
        expected = np.array([
            [37, 1, 1, 25], 
            [ 4, 0, 0,  8], 
            [22, 2, 2, 42]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_vertical_lines(self):
        image = np.array([
            [1, 2, 3], 
            [1, 2, 3]], dtype=np.uint8)
        actual = ImageShape2x2.apply(image)
        a = ImageShape2x2.TOPLEFT_EQUAL_BOTTOMLEFT | ImageShape2x2.TOPRIGHT_EQUAL_BOTTOMRIGHT
        self.assertEqual(a, 12)
        expected = np.array([
            [37, 1, 1, 25], 
            [ a, a, a,  a], 
            [22, 2, 2, 42]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_horizontal_lines(self):
        image = np.array([
            [1, 1, 1], 
            [2, 2, 2]], dtype=np.uint8)
        actual = ImageShape2x2.apply(image)
        a = ImageShape2x2.TOPLEFT_EQUAL_TOPRIGHT | ImageShape2x2.BOTTOMLEFT_EQUAL_BOTTOMRIGHT
        self.assertEqual(a, 3)
        expected = np.array([
            [37, a, a, 25], 
            [ 4, a, a,  8], 
            [22, a, a, 42]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_diagonal_lines_topleft_bottomright(self):
        image = np.array([
            [1, 4, 5, 6], 
            [2, 1, 4, 5],
            [3, 2, 1, 4]], dtype=np.uint8)
        actual = ImageShape2x2.apply(image)
        a = ImageShape2x2.TOPLEFT_EQUAL_BOTTOMRIGHT
        self.assertEqual(a, 16)
        expected = np.array([
            [37, 1, 1, 1, 25], 
            [ 4, a, a, a,  8], 
            [ 4, a, a, a,  8], 
            [22, 2, 2, 2, 42]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_diagonal_lines_topright_bottomleft(self):
        image = np.array([
            [3, 2, 1, 4],
            [2, 1, 4, 5],
            [1, 4, 5, 6]], dtype=np.uint8)
        actual = ImageShape2x2.apply(image)
        a = ImageShape2x2.TOPRIGHT_EQUAL_BOTTOMLEFT
        self.assertEqual(a, 32)
        expected = np.array([
            [37, 1, 1, 1, 25], 
            [ 4, a, a, a,  8], 
            [ 4, a, a, a,  8], 
            [22, 2, 2, 2, 42]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_all_same_color(self):
        image = np.array([
            [9, 9, 9, 9], 
            [9, 9, 9, 9], 
            [9, 9, 9, 9]], dtype=np.uint8)
        actual = ImageShape2x2.apply(image)
        a = 63
        expected = np.array([
            [37, 3, 3, 3, 25], 
            [12, a, a, a, 12], 
            [12, a, a, a, 12], 
            [22, 3, 3, 3, 42]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_shape_id_list_all_1x1(self):
        image = np.array([[9]], dtype=np.uint8)
        actual = ImageShape2x2.shape_id_list(image)
        expected = [22, 25, 37, 42]
        self.assertTrue(np.array_equal(actual, expected))

    def test_shape_id_list_all_rect_with_one_color(self):
        image = np.array([
            [9, 9, 9, 9], 
            [9, 9, 9, 9], 
            [9, 9, 9, 9]], dtype=np.uint8)
        actual = ImageShape2x2.shape_id_list(image)
        expected = [3, 12, 22, 25, 37, 42, 63]
        self.assertTrue(np.array_equal(actual, expected))

    def test_shape_id_list_all_diagonal_topleftbottomright(self):
        image = np.array([
            [9, 7, 9, 9, 9], 
            [9, 9, 7, 9, 9], 
            [9, 9, 9, 7, 9]], dtype=np.uint8)
        actual = ImageShape2x2.shape_id_list(image)
        expected = [1, 2, 3, 12, 22, 25, 37, 42, 48, 63]
        self.assertTrue(np.array_equal(actual, expected))

    def test_shape_id_list_all_diagonal_toprightbottomleft(self):
        image = np.array([
            [9, 9, 9, 7, 9],
            [9, 9, 7, 9, 9], 
            [9, 7, 9, 9, 9]], dtype=np.uint8) 
        actual = ImageShape2x2.shape_id_list(image)
        expected = [1, 2, 3, 12, 22, 25, 37, 42, 48, 63]
        self.assertTrue(np.array_equal(actual, expected))

    def test_shape_id_list_all_diagonal_alternating_lines(self):
        image = np.array([
            [9, 9, 9, 9, 9],
            [7, 7, 7, 7, 7],
            [9, 9, 9, 9, 9],
            [7, 7, 7, 7, 7]], dtype=np.uint8) 
        actual = ImageShape2x2.shape_id_list(image)
        expected = [3, 4, 8, 22, 25, 37, 42]
        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
