import unittest
import numpy as np
from .image_compress import *

class TestImageCompress(unittest.TestCase):
    def test_compress_x_1(self):
        image = np.array([
            [3, 7, 3], 
            [3, 7, 3], 
            [3, 7, 3]], dtype=np.uint8)
        actual = image_compress_x(image)
        expected = np.array([
            [3, 7, 3], 
            [3, 7, 3], 
            [3, 7, 3]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_compress_x_2(self):
        image = np.array([
            [5, 3, 3, 2, 2, 3], 
            [3, 3, 3, 7, 7, 3], 
            [2, 2, 2, 3, 3, 3]], dtype=np.uint8)
        actual = image_compress_x(image)
        expected = np.array([
            [5, 3, 2, 3], 
            [3, 3, 7, 3], 
            [2, 2, 3, 3]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_compress_x_3(self):
        image = np.array([
            [1, 2, 2, 2, 3, 4, 4], 
            [7, 7, 7, 7, 7, 7, 7], 
            [1, 2, 2, 2, 3, 4, 4]], dtype=np.uint8)
        actual = image_compress_x(image)
        expected = np.array([
            [1, 2, 3, 4], 
            [7, 7, 7, 7], 
            [1, 2, 3, 4]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_compress_y_1(self):
        image = np.array([
            [1, 2], 
            [2, 2], 
            [3, 2]], dtype=np.uint8)
        actual = image_compress_y(image)
        expected = np.array([
            [1, 2], 
            [2, 2], 
            [3, 2]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_compress_y_2(self):
        image = np.array([
            [1, 2], 
            [1, 2], 
            [2, 2], 
            [2, 2], 
            [3, 2], 
            [3, 2]], dtype=np.uint8)
        actual = image_compress_y(image)
        expected = np.array([
            [1, 2], 
            [2, 2], 
            [3, 2]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_compress_xy_1(self):
        image = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]], dtype=np.uint8)
        actual = image_compress_xy(image)
        expected = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_compress_xy_1(self):
        image = np.array([
            [1, 1, 1, 2, 3, 3], 
            [1, 1, 1, 2, 3, 3], 
            [4, 4, 4, 5, 6, 6], 
            [4, 4, 4, 5, 6, 6], 
            [4, 4, 4, 5, 6, 6], 
            [7, 7, 7, 8, 9, 9]], dtype=np.uint8)
        actual = image_compress_xy(image)
        expected = np.array([
            [1, 2, 3], 
            [4, 5, 6], 
            [7, 8, 9]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
