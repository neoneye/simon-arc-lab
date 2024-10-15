import unittest
import numpy as np
from .image_mix import *

class TestImageMix(unittest.TestCase):
    def test_success_zero_and_one(self):
        mask = np.array([
            [0, 1, 0, 1], 
            [1, 0, 1, 0],
            [0, 1, 0, 1]], dtype=np.uint8)
        image0 = np.array([
            [5, 5, 5, 5], 
            [6, 6, 6, 6],
            [5, 5, 5, 5]], dtype=np.uint8)
        image1 = np.array([
            [7, 8, 7, 8], 
            [7, 8, 7, 8],
            [7, 8, 7, 8]], dtype=np.uint8)
        actual = image_mix(mask, image0, image1)
        expected = np.array([
            [5, 8, 5, 8], 
            [7, 6, 7, 6], 
            [5, 8, 5, 8]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_success_value_bigger_than_one(self):
        mask = np.array([
            [0, 3, 0, 2], 
            [255, 0, 99, 0],
            [0, 1, 0, 128]], dtype=np.uint8)
        image0 = np.array([
            [5, 5, 5, 5], 
            [6, 6, 6, 6],
            [5, 5, 5, 5]], dtype=np.uint8)
        image1 = np.array([
            [7, 8, 7, 8], 
            [7, 8, 7, 8],
            [7, 8, 7, 8]], dtype=np.uint8)
        actual = image_mix(mask, image0, image1)
        expected = np.array([
            [5, 8, 5, 8], 
            [7, 6, 7, 6], 
            [5, 8, 5, 8]], dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_size_mismatch0(self):
        mask = np.array([
            [0, 1, 0], 
            [1, 0, 1],
            [0, 1, 0]], dtype=np.uint8)
        image0 = np.array([
            [5, 5, 5, 5], 
            [6, 6, 6, 6],
            [5, 5, 5, 5]], dtype=np.uint8)
        image1 = np.array([
            [7, 8, 7, 8], 
            [7, 8, 7, 8],
            [7, 8, 7, 8]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            image_mix(mask, image0, image1)

    def test_size_mismatch1(self):
        mask = np.array([
            [0, 1, 0, 1], 
            [1, 0, 1, 0],
            [0, 1, 0, 1]], dtype=np.uint8)
        image0 = np.array([
            [5, 5, 5, 5], 
            [5, 5, 5, 5]], dtype=np.uint8)
        image1 = np.array([
            [7, 8, 7, 8], 
            [7, 8, 7, 8],
            [7, 8, 7, 8]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            image_mix(mask, image0, image1)

    def test_size_mismatch2(self):
        mask = np.array([
            [0, 1, 1], 
            [1, 0, 0],
            [0, 0, 1]], dtype=np.uint8)
        image0 = np.array([
            [5, 5, 5, 5], 
            [5, 5, 5, 5]], dtype=np.uint8)
        image1 = np.array([
            [7, 8, 7, 8], 
            [7, 8, 7, 8],
            [7, 8, 7, 8]], dtype=np.uint8)
        with self.assertRaises(ValueError):
            image_mix(mask, image0, image1)

if __name__ == '__main__':
    unittest.main()
