import unittest
import numpy as np
from .image_shape3x3_opposite import *

class TestImageShape3x3Opposite(unittest.TestCase):
    def test_10000_topleft_bottomright(self):
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
        np.testing.assert_array_equal(actual, expected)

    def test_10001_topcenter_bottomcenter(self):
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
        np.testing.assert_array_equal(actual, expected)

    def test_10002_topright_bottomleft(self):
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
        np.testing.assert_array_equal(actual, expected)

    def test_10003_centerleft_centerright(self):
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
        np.testing.assert_array_equal(actual, expected)

    def test_10004_all_satisfied(self):
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
        np.testing.assert_array_equal(actual, expected)

    def test_20000_shape_id_list(self):
        image = np.array([
            [1, 1, 2, 2, 1],
            [1, 0, 0, 0, 1],
            [2, 0, 5, 0, 2],
            [2, 0, 0, 0, 2],
            [1, 1, 2, 2, 1]], dtype=np.uint8) 
        actual = ImageShape3x3Opposite.shape_id_list(image)
        expected = [0, 1, 2, 4, 8, 15]
        self.assertEqual(actual, expected)

    def test_20001_shape_id_list(self):
        # Arrange
        image = np.array([
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [5, 6, 5, 6]], dtype=np.uint8) 
        # Act
        actual = ImageShape3x3Opposite.shape_id_list(image)
        # Assert
        value = ImageShape3x3Opposite.CENTERLEFT_CENTERRIGHT
        expected = [0, value]
        self.assertEqual(actual, expected)

    def test_20002_shape_id_list(self):
        # Arrange
        image = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [1, 2, 3, 4]], dtype=np.uint8) 
        # Act
        actual = ImageShape3x3Opposite.shape_id_list(image)
        # Assert
        value = ImageShape3x3Opposite.TOPCENTER_BOTTOMCENTER
        expected = [0, value]
        self.assertEqual(actual, expected)

    def test_20003_shape_id_list(self):
        # Arrange
        image = np.array([
            [9, 0, 1, 2],
            [5, 6, 7, 8],
            [1, 2, 3, 4]], dtype=np.uint8) 
        # Act
        actual = ImageShape3x3Opposite.shape_id_list(image)
        # Assert
        value = ImageShape3x3Opposite.TOPRIGHT_BOTTOMLEFT
        expected = [0, value]
        self.assertEqual(actual, expected)

    def test_20004_shape_id_list(self):
        # Arrange
        image = np.array([
            [1, 2, 3, 4], 
            [5, 6, 7, 8],
            [9, 0, 1, 2]], dtype=np.uint8)
        # Act
        actual = ImageShape3x3Opposite.shape_id_list(image)
        # Assert
        value = ImageShape3x3Opposite.TOPLEFT_BOTTOMRIGHT
        expected = [0, value]
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
