import unittest
import numpy as np
from .image_shape3x3_opposite import *

class TestImageShape3x3Opposite(unittest.TestCase):
    def test_topleft_bottomright(self):
        image = np.array([
            [1, 2, 3, 4], 
            [5, 6, 7, 8],
            [9, 0, 1, 2]], dtype=np.uint8)
        actual = ImageShape3x3Opposite.apply(image)
        a = ImageShape3x3Opposite.TOPLEFT_BOTTOMRIGHT
        expected = np.array([
            [0, 0, 0, 0], 
            [0, a, a, 0], 
            [0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_topcenter_bottomcenter(self):
        image = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 2, 3, 4]], dtype=np.uint8) 
        actual = ImageShape3x3Opposite.apply(image)
        a = ImageShape3x3Opposite.TOPCENTER_BOTTOMCENTER
        expected = np.array([
            [0, 0, 0, 0], 
            [a, a, a, a], 
            [0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_topright_bottomleft(self):
        image = np.array([
            [9, 0, 1, 2],
            [5, 6, 7, 8],
            [1, 2, 3, 4]], dtype=np.uint8) 
        actual = ImageShape3x3Opposite.apply(image)
        a = ImageShape3x3Opposite.TOPRIGHT_BOTTOMLEFT
        expected = np.array([
            [0, 0, 0, 0], 
            [0, a, a, 0], 
            [0, 0, 0, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_centerleft_centerright(self):
        image = np.array([
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [5, 6, 5, 6]], dtype=np.uint8) 
        actual = ImageShape3x3Opposite.apply(image)
        a = ImageShape3x3Opposite.CENTERLEFT_CENTERRIGHT
        expected = np.array([
            [0, a, a, 0], 
            [0, a, a, 0], 
            [0, a, a, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(actual, expected))

    def test_all_satisfied(self):
        image = np.array([
            [1, 1, 2, 2, 1],
            [1, 0, 0, 0, 1],
            [2, 0, 5, 0, 2],
            [2, 0, 0, 0, 2],
            [1, 1, 2, 2, 1]], dtype=np.uint8) 
        actual = ImageShape3x3Opposite.apply(image)
        a = ImageShape3x3Opposite.TOPLEFT_BOTTOMRIGHT | ImageShape3x3Opposite.TOPCENTER_BOTTOMCENTER | ImageShape3x3Opposite.TOPRIGHT_BOTTOMLEFT | ImageShape3x3Opposite.CENTERLEFT_CENTERRIGHT
        self.assertEqual(a, 0x0f)
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 4, 8, 1, 0],
            [0, 2, a, 2, 0],
            [0, 1, 8, 4, 0],
            [0, 0, 0, 0, 0]], dtype=np.uint8) 
        self.assertTrue(np.array_equal(actual, expected))

if __name__ == '__main__':
    unittest.main()
