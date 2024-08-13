import unittest
import numpy as np
from .image_symmetry import *

class TestImageSymmetry(unittest.TestCase):
    def test_10000_hstack2(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        output, instruction_sequence = ImageSymmetry(image).execute('hstack2')
        # Assert
        expected = np.array([
            [1, 2, 3, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(orig orig)')

    def test_10001_hstack3(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        output, instruction_sequence = ImageSymmetry(image).execute('hstack3')
        # Assert
        expected = np.array([
            [1, 2, 3, 1, 2, 3, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(orig orig orig)')

    def test_10002_vstack2(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        output, instruction_sequence = ImageSymmetry(image).execute('vstack2')
        # Assert
        expected = np.array([
            [1, 2, 3], 
            [1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'vstack(orig orig)')

    def test_10003_vstack3(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        output, instruction_sequence = ImageSymmetry(image).execute('vstack3')
        # Assert
        expected = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'vstack(orig orig orig)')

    def test_10004_2x2(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        output, instruction_sequence = ImageSymmetry(image).execute('2x2')
        # Assert
        expected = np.array([
            [1, 2, 3, 1, 2, 3],
            [1, 2, 3, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, '2x2(orig orig orig orig)')

    def test_20000_name_flipx(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        i = ImageSymmetry(image)
        i.use_flipx_for_name_image_index(0)
        output, instruction_sequence = i.execute('hstack2')
        # Assert
        expected = np.array([
            [3, 2, 1, 1, 2, 3],
            [6, 5, 4, 4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(flipx orig)')

    def test_20001_name_flipy(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        i = ImageSymmetry(image)
        i.use_flipy_for_name_image_index(1)
        output, instruction_sequence = i.execute('hstack2')
        # Assert
        expected = np.array([
            [1, 2, 3, 4, 5, 6],
            [4, 5, 6, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(orig flipy)')

    def test_20002_name_180(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        i = ImageSymmetry(image)
        i.use_180_for_name_image_index(1)
        output, instruction_sequence = i.execute('hstack2')
        # Assert
        expected = np.array([
            [1, 2, 3, 6, 5, 4],
            [4, 5, 6, 3, 2, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(orig 180)')

if __name__ == '__main__':
    unittest.main()
