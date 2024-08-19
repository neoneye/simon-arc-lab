import unittest
import numpy as np
from .image_symmetry import *

class TestImageSymmetry(unittest.TestCase):
    def test_10000_rect_hstack2(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetryRect(ImageSymmetryPatternId.HSTACK2)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(orig orig)')

    def test_10001_rect_hstack3(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetryRect(ImageSymmetryPatternId.HSTACK3)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3, 1, 2, 3, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(orig orig orig)')

    def test_10002_rect_hstack4(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetryRect(ImageSymmetryPatternId.HSTACK4)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(orig orig orig orig)')

    def test_10003_rect_hstack5(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetryRect(ImageSymmetryPatternId.HSTACK5)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(orig orig orig orig orig)')

    def test_10004_rect_vstack2(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetryRect(ImageSymmetryPatternId.VSTACK2)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3], 
            [1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'vstack(orig orig)')

    def test_10005_rect_vstack3(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetryRect(ImageSymmetryPatternId.VSTACK3)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'vstack(orig orig orig)')

    def test_10006_rect_vstack4(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetryRect(ImageSymmetryPatternId.VSTACK4)
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

    def test_10007_rect_vstack5(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetryRect(ImageSymmetryPatternId.VSTACK5)
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

    def test_10008_rect_grid2x2(self):
        # Arrange
        image = np.array([
            [1, 2, 3]], dtype=np.uint8)
        # Act
        i = ImageSymmetryRect(ImageSymmetryPatternId.GRID2X2)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3, 1, 2, 3],
            [1, 2, 3, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, '2x2(orig orig orig orig)')

    def test_20000_rect_name_flipx(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        i = ImageSymmetryRect(ImageSymmetryPatternId.HSTACK2)
        i.use_flipx_for_index(0)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [3, 2, 1, 1, 2, 3],
            [6, 5, 4, 4, 5, 6]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(flipx orig)')

    def test_20001_rect_name_flipy(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        i = ImageSymmetryRect(ImageSymmetryPatternId.HSTACK2)
        i.use_flipy_for_index(1)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3, 4, 5, 6],
            [4, 5, 6, 1, 2, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(orig flipy)')

    def test_20002_rect_name_180(self):
        # Arrange
        image = np.array([
            [1, 2, 3],
            [4, 5, 6]], dtype=np.uint8)
        # Act
        i = ImageSymmetryRect(ImageSymmetryPatternId.HSTACK2)
        i.use_180_for_index(1)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 2, 3, 6, 5, 4],
            [4, 5, 6, 3, 2, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(instruction_sequence, 'hstack(orig 180)')

    def test_30000_rect_execute_with_multiple_different_images(self):
        # Arrange
        i = ImageSymmetryRect(ImageSymmetryPatternId.HSTACK2)
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

    def test_40000_square_rotate_cw(self):
        # Arrange
        image = np.array([
            [1, 2],
            [3, 4]], dtype=np.uint8)
        # Act
        i = ImageSymmetrySquare(ImageSymmetryPatternId.HSTACK2)
        i.use_rotate_cw_for_index(0)
        i.use_rotate_cw_for_index(1)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [3, 1, 3, 1],
            [4, 2, 4, 2]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(instruction_sequence, 'hstack(cw cw)')

    def test_40001_square_rotate_ccw(self):
        # Arrange
        image = np.array([
            [1, 2],
            [3, 4]], dtype=np.uint8)
        # Act
        i = ImageSymmetrySquare(ImageSymmetryPatternId.HSTACK2)
        i.use_rotate_ccw_for_index(0)
        i.use_rotate_ccw_for_index(1)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [2, 4, 2, 4],
            [1, 3, 1, 3]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(instruction_sequence, 'hstack(ccw ccw)')

    def test_40002_square_flip_diagonal_a(self):
        # Arrange
        image = np.array([
            [1, 2],
            [3, 4]], dtype=np.uint8)
        # Act
        i = ImageSymmetrySquare(ImageSymmetryPatternId.HSTACK2)
        i.use_flipa_for_index(0)
        i.use_flipa_for_index(1)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [1, 3, 1, 3],
            [2, 4, 2, 4]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(instruction_sequence, 'hstack(flipa flipa)')

    def test_40003_square_flip_diagonal_b(self):
        # Arrange
        image = np.array([
            [1, 2],
            [3, 4]], dtype=np.uint8)
        # Act
        i = ImageSymmetrySquare(ImageSymmetryPatternId.HSTACK2)
        i.use_flipb_for_index(0)
        i.use_flipb_for_index(1)
        output = i.execute(image)
        instruction_sequence = i.instruction_sequence()
        # Assert
        expected = np.array([
            [4, 2, 4, 2],
            [3, 1, 3, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(instruction_sequence, 'hstack(flipb flipb)')

if __name__ == '__main__':
    unittest.main()
