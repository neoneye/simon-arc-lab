import unittest
import numpy as np
from .image_symmetry import *

class TestImageSymmetry(unittest.TestCase):
    def test_10000_hstack2(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetry(ImageSymmetryPatternId.HSTACK2)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
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
        i = ImageSymmetry(ImageSymmetryPatternId.HSTACK3)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3, 1, 2, 3, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(orig orig orig)')

    def test_10002_hstack4(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetry(ImageSymmetryPatternId.HSTACK4)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(orig orig orig orig)')

    def test_10003_hstack5(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetry(ImageSymmetryPatternId.HSTACK5)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(orig orig orig orig orig)')

    def test_10004_vstack2(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetry(ImageSymmetryPatternId.VSTACK2)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3], 
            [1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'vstack(orig orig)')

    def test_10005_vstack3(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetry(ImageSymmetryPatternId.VSTACK3)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'vstack(orig orig orig)')

    def test_10006_vstack4(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetry(ImageSymmetryPatternId.VSTACK4)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'vstack(orig orig orig orig)')

    def test_10007_vstack5(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetry(ImageSymmetryPatternId.VSTACK5)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'vstack(orig orig orig orig orig)')

    def test_10008_grid2x2(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetry(ImageSymmetryPatternId.GRID2X2)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
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
        i = ImageSymmetry(ImageSymmetryPatternId.HSTACK2)
        i.use_flipx_for_index(0)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
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
        i = ImageSymmetry(ImageSymmetryPatternId.HSTACK2)
        i.use_flipy_for_index(1)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
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
        i = ImageSymmetry(ImageSymmetryPatternId.HSTACK2)
        i.use_180_for_index(1)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3, 6, 5, 4],
            [4, 5, 6, 3, 2, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(orig 180)')

    def test_30000_execute_with_multiple_different_images(self):
        # Arrange
        i = ImageSymmetry(ImageSymmetryPatternId.HSTACK2)
        i.use_flipx_for_index(0)
        image123 = np.array([
            [1, 2, 3]], dtype=np.uint8)
        image456 = np.array([
            [4, 5, 6]], dtype=np.uint8)
        # Act
        output123 = i.execute(image123)
        output456 = i.execute(image456)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected123 = np.array([
            [3, 2, 1, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output123, expected123)
        expected456 = np.array([
            [6, 5, 4, 4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(output456, expected456)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(flipx orig)')

if __name__ == '__main__':
    unittest.main()
